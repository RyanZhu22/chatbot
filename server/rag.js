import { promises as fs } from 'fs'
import path from 'path'

const RAG_ENABLED = String(process.env.RAG_ENABLED || 'true').toLowerCase() !== 'false'
const RAG_KNOWLEDGE_DIR = process.env.RAG_KNOWLEDGE_DIR || 'knowledge'
const RAG_TOP_K = Math.max(1, Number(process.env.RAG_TOP_K || 4))
const RAG_MIN_SCORE = Math.max(0, Number(process.env.RAG_MIN_SCORE || 0.8))
const RAG_CHUNK_SIZE = Math.max(200, Number(process.env.RAG_CHUNK_SIZE || 800))
const RAG_CONTEXT_MAX_CHARS = Math.max(800, Number(process.env.RAG_CONTEXT_MAX_CHARS || 4000))
const RAG_CACHE_TTL_MS = Math.max(1000, Number(process.env.RAG_CACHE_TTL_MS || 15000))
const RAG_BM25_K1 = Math.max(0.2, Number(process.env.RAG_BM25_K1 || 1.2))
const RAG_BM25_B = Math.min(1, Math.max(0, Number(process.env.RAG_BM25_B || 0.75)))
const RAG_CANDIDATE_MULTIPLIER = Math.max(2, Number(process.env.RAG_CANDIDATE_MULTIPLIER || 8))
const RAG_MMR_LAMBDA = Math.min(1, Math.max(0, Number(process.env.RAG_MMR_LAMBDA || 0.78)))
const RAG_NGRAM_SIZE = Math.max(2, Number(process.env.RAG_NGRAM_SIZE || 3))

const SUPPORTED_EXTENSIONS = new Set(['.txt', '.md', '.markdown'])
const knowledgeRoot = path.resolve(process.cwd(), RAG_KNOWLEDGE_DIR)

let cache = {
  loadedAt: 0,
  chunks: [],
  files: 0,
  lastError: null,
  dfMap: new Map(),
  avgTokens: 0,
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value))
}

function intersectCount(setA, setB) {
  if (!setA.size || !setB.size) {
    return 0
  }
  let count = 0
  const smaller = setA.size <= setB.size ? setA : setB
  const larger = smaller === setA ? setB : setA
  for (const item of smaller) {
    if (larger.has(item)) {
      count += 1
    }
  }
  return count
}

function buildCharNgrams(text, size = RAG_NGRAM_SIZE) {
  const normalized = String(text || '').toLowerCase().replace(/\s+/g, '')
  const result = new Set()
  if (!normalized) {
    return result
  }
  if (normalized.length <= size) {
    result.add(normalized)
    return result
  }
  for (let index = 0; index <= normalized.length - size; index += 1) {
    result.add(normalized.slice(index, index + size))
  }
  return result
}

function tokenize(text) {
  const safeText = String(text || '').toLowerCase()
  const tokens = []
  const english = safeText.match(/[a-z0-9_]{2,}/g) || []
  tokens.push(...english)

  const chineseChars = safeText.match(/[\u4e00-\u9fff]/g) || []
  tokens.push(...chineseChars)

  const chineseBlocks = safeText.match(/[\u4e00-\u9fff]{2,}/g) || []
  for (const block of chineseBlocks) {
    if (block.length <= 8) {
      tokens.push(block)
    }
    for (let index = 0; index < block.length - 1; index += 1) {
      tokens.push(block.slice(index, index + 2))
    }
  }

  return tokens
}

function tokenWeight(token) {
  if (/^[\u4e00-\u9fff]{2,}$/.test(token)) {
    return 1.35
  }
  if (/^[\u4e00-\u9fff]$/.test(token)) {
    return 0.8
  }
  if (token.length >= 6) {
    return 1.25
  }
  return 1
}

function buildQueryFeatures(queryText) {
  const normalized = String(queryText || '').replace(/\s+/g, ' ').trim().toLowerCase()
  const tokens = tokenize(normalized)
  const uniqueTokens = [...new Set(tokens)]
  const ngrams = buildCharNgrams(normalized, RAG_NGRAM_SIZE)

  const phrases = []
  if (normalized.length >= 8) {
    phrases.push(normalized)
  }
  const cjkPhrases = normalized.match(/[\u4e00-\u9fff]{3,}/g) || []
  for (const phrase of cjkPhrases) {
    phrases.push(phrase)
  }
  const longWords = normalized.match(/[a-z0-9_]{5,}/g) || []
  for (const word of longWords) {
    phrases.push(word)
  }

  return {
    text: normalized,
    uniqueTokens,
    ngrams,
    phrases: [...new Set(phrases)].slice(0, 10),
  }
}

function createChunk(source, text, index) {
  const normalized = String(text || '').replace(/\s+/g, ' ').trim()
  if (!normalized) {
    return null
  }
  const tokens = tokenize(normalized)
  if (tokens.length === 0) {
    return null
  }
  const tokenFreq = new Map()
  for (const token of tokens) {
    tokenFreq.set(token, (tokenFreq.get(token) || 0) + 1)
  }
  return {
    id: `${source}#${index}`,
    source,
    text: normalized,
    tokensCount: tokens.length,
    tokenFreq,
    tokenSet: new Set(tokenFreq.keys()),
    ngramSet: buildCharNgrams(normalized, RAG_NGRAM_SIZE),
    lowerText: normalized.toLowerCase(),
  }
}

function splitIntoChunks(text, maxChars) {
  const paragraphs = String(text || '')
    .replace(/\r\n/g, '\n')
    .split(/\n{2,}/)
    .map((item) => item.trim())
    .filter(Boolean)

  const result = []
  let current = ''

  const pushCurrent = () => {
    const trimmed = current.trim()
    if (trimmed) {
      result.push(trimmed)
    }
    current = ''
  }

  for (const paragraph of paragraphs) {
    if (paragraph.length > maxChars) {
      const sentences = paragraph.split(/(?<=[。！？.!?])\s*/).filter(Boolean)
      for (const sentence of sentences) {
        if ((current + ' ' + sentence).trim().length <= maxChars) {
          current = `${current} ${sentence}`.trim()
          continue
        }
        pushCurrent()
        if (sentence.length <= maxChars) {
          current = sentence
          continue
        }
        for (let start = 0; start < sentence.length; start += maxChars) {
          result.push(sentence.slice(start, start + maxChars))
        }
      }
      continue
    }

    if ((current + '\n\n' + paragraph).trim().length <= maxChars) {
      current = `${current}\n\n${paragraph}`.trim()
    } else {
      pushCurrent()
      current = paragraph
    }
  }

  pushCurrent()
  return result
}

async function walkFiles(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  const files = []
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      const nested = await walkFiles(fullPath)
      files.push(...nested)
      continue
    }
    const ext = path.extname(entry.name).toLowerCase()
    if (SUPPORTED_EXTENSIONS.has(ext)) {
      files.push(fullPath)
    }
  }
  return files
}

function buildIndexStats(chunks) {
  const dfMap = new Map()
  let totalTokens = 0
  for (const chunk of chunks) {
    totalTokens += chunk.tokensCount
    for (const token of chunk.tokenSet) {
      dfMap.set(token, (dfMap.get(token) || 0) + 1)
    }
  }
  const avgTokens = chunks.length > 0 ? totalTokens / chunks.length : 0
  return { dfMap, avgTokens }
}

async function rebuildCache() {
  if (!RAG_ENABLED) {
    cache = {
      loadedAt: Date.now(),
      chunks: [],
      files: 0,
      lastError: null,
      dfMap: new Map(),
      avgTokens: 0,
    }
    return cache
  }

  const rootExists = await fs
    .access(knowledgeRoot)
    .then(() => true)
    .catch(() => false)

  if (!rootExists) {
    cache = {
      loadedAt: Date.now(),
      chunks: [],
      files: 0,
      lastError: `knowledge directory not found: ${knowledgeRoot}`,
      dfMap: new Map(),
      avgTokens: 0,
    }
    return cache
  }

  const files = await walkFiles(knowledgeRoot)
  const chunks = []

  for (const filePath of files) {
    const content = await fs.readFile(filePath, 'utf8').catch(() => '')
    const relative = path.relative(process.cwd(), filePath).replaceAll('\\', '/')
    const parts = splitIntoChunks(content, RAG_CHUNK_SIZE)
    parts.forEach((part, index) => {
      const chunk = createChunk(relative, part, index)
      if (chunk) {
        chunks.push(chunk)
      }
    })
  }

  const stats = buildIndexStats(chunks)
  cache = {
    loadedAt: Date.now(),
    chunks,
    files: files.length,
    lastError: null,
    dfMap: stats.dfMap,
    avgTokens: stats.avgTokens,
  }
  return cache
}

async function ensureCache() {
  const now = Date.now()
  if (now - cache.loadedAt < RAG_CACHE_TTL_MS) {
    return cache
  }
  return rebuildCache()
}

function bm25Score(chunk, queryFeatures, state) {
  const totalDocs = Math.max(1, state.chunks.length)
  const avgLen = Math.max(1, state.avgTokens)
  let score = 0

  for (const token of queryFeatures.uniqueTokens) {
    const tf = chunk.tokenFreq.get(token) || 0
    if (tf <= 0) {
      continue
    }
    const df = state.dfMap.get(token) || 0
    const idf = Math.log(1 + (totalDocs - df + 0.5) / (df + 0.5))
    const numerator = tf * (RAG_BM25_K1 + 1)
    const denominator =
      tf + RAG_BM25_K1 * (1 - RAG_BM25_B + (RAG_BM25_B * chunk.tokensCount) / avgLen)
    score += (idf * numerator * tokenWeight(token)) / Math.max(denominator, 1e-9)
  }

  return score
}

function phraseScore(chunk, queryFeatures) {
  let score = 0
  for (const phrase of queryFeatures.phrases) {
    if (phrase.length < 3) {
      continue
    }
    if (chunk.lowerText.includes(phrase)) {
      score += phrase.length >= 8 ? 1.5 : 0.7
    }
  }
  if (queryFeatures.text.length >= 8 && chunk.lowerText.includes(queryFeatures.text)) {
    score += 2
  }
  return score
}

function ngramSimilarity(chunk, queryFeatures) {
  if (!queryFeatures.ngrams.size || !chunk.ngramSet.size) {
    return 0
  }
  const overlap = intersectCount(queryFeatures.ngrams, chunk.ngramSet)
  if (!overlap) {
    return 0
  }
  return overlap / Math.sqrt(queryFeatures.ngrams.size * chunk.ngramSet.size)
}

function queryCoverage(chunk, queryFeatures) {
  if (!queryFeatures.uniqueTokens.length) {
    return 0
  }
  let matched = 0
  for (const token of queryFeatures.uniqueTokens) {
    if (chunk.tokenSet.has(token)) {
      matched += 1
    }
  }
  return matched / queryFeatures.uniqueTokens.length
}

function scoreChunk(chunk, queryFeatures, state) {
  const bm25 = bm25Score(chunk, queryFeatures, state)
  const phrase = phraseScore(chunk, queryFeatures)
  const ngram = ngramSimilarity(chunk, queryFeatures)
  const coverage = queryCoverage(chunk, queryFeatures)
  const score = bm25 + phrase + ngram * 2.2 + coverage * 1.2
  return {
    score,
    metrics: {
      bm25: Number(bm25.toFixed(4)),
      phrase: Number(phrase.toFixed(4)),
      ngram: Number(ngram.toFixed(4)),
      coverage: Number(coverage.toFixed(4)),
    },
  }
}

function chunkSimilarity(left, right) {
  const tokenOverlap = left.tokenSet.size && right.tokenSet.size
    ? intersectCount(left.tokenSet, right.tokenSet) / Math.max(left.tokenSet.size, right.tokenSet.size)
    : 0
  const ngramOverlap = left.ngramSet.size && right.ngramSet.size
    ? intersectCount(left.ngramSet, right.ngramSet) / Math.max(left.ngramSet.size, right.ngramSet.size)
    : 0
  return clamp01(tokenOverlap * 0.55 + ngramOverlap * 0.45)
}

function applyMMR(candidates, topK) {
  if (candidates.length <= topK) {
    return candidates
  }

  const maxRelevance = Math.max(...candidates.map((item) => item.score), 1)
  const selected = []
  const remaining = [...candidates]

  while (selected.length < topK && remaining.length > 0) {
    let bestIndex = 0
    let bestScore = -Infinity

    for (let index = 0; index < remaining.length; index += 1) {
      const candidate = remaining[index]
      const relevance = candidate.score / maxRelevance
      let redundancy = 0
      for (const chosen of selected) {
        redundancy = Math.max(redundancy, chunkSimilarity(candidate, chosen))
      }
      const mmrScore = RAG_MMR_LAMBDA * relevance - (1 - RAG_MMR_LAMBDA) * redundancy
      if (mmrScore > bestScore) {
        bestScore = mmrScore
        bestIndex = index
      }
    }

    selected.push(remaining[bestIndex])
    remaining.splice(bestIndex, 1)
  }

  return selected.sort((a, b) => b.score - a.score)
}

function buildContextMessage(matches) {
  if (!matches.length) {
    return null
  }

  const sections = []
  let consumed = 0

  for (let index = 0; index < matches.length; index += 1) {
    const item = matches[index]
    const header = `[${index + 1}] Source: ${item.source}\n`
    const body = item.text
    const block = `${header}${body}\n`
    if (consumed + block.length > RAG_CONTEXT_MAX_CHARS) {
      break
    }
    sections.push(block)
    consumed += block.length
  }

  if (sections.length === 0) {
    return null
  }

  return [
    'You have access to internal knowledge snippets below.',
    'Use them when relevant. If used, cite source path in plain text like [knowledge/...].',
    '',
    ...sections,
  ].join('\n')
}

export async function retrieveKnowledge(query) {
  const safeQuery = String(query || '').trim()
  if (!RAG_ENABLED || !safeQuery) {
    return {
      matches: [],
      citations: [],
      contextMessage: null,
    }
  }

  const state = await ensureCache()
  if (!state.chunks.length) {
    return {
      matches: [],
      citations: [],
      contextMessage: null,
    }
  }

  const queryFeatures = buildQueryFeatures(safeQuery)
  if (!queryFeatures.uniqueTokens.length) {
    return {
      matches: [],
      citations: [],
      contextMessage: null,
    }
  }

  const candidateLimit = Math.min(
    state.chunks.length,
    Math.max(RAG_TOP_K, RAG_TOP_K * RAG_CANDIDATE_MULTIPLIER),
  )

  const scored = state.chunks
    .map((chunk) => {
      const result = scoreChunk(chunk, queryFeatures, state)
      return {
        ...chunk,
        score: result.score,
        metrics: result.metrics,
      }
    })
    .filter((item) => item.score >= RAG_MIN_SCORE)
    .sort((a, b) => b.score - a.score)
    .slice(0, candidateLimit)

  const ranked = applyMMR(scored, RAG_TOP_K)

  const citations = []
  const seen = new Set()
  for (const item of ranked) {
    if (seen.has(item.source)) {
      continue
    }
    seen.add(item.source)
    citations.push({ source: item.source })
  }

  return {
    matches: ranked.map((item) => ({
      source: item.source,
      text: item.text,
      score: Number(item.score.toFixed(4)),
      metrics: item.metrics,
    })),
    citations,
    contextMessage: buildContextMessage(ranked),
  }
}

export async function getRagStatus() {
  const state = await ensureCache()
  return {
    enabled: RAG_ENABLED,
    retriever: 'hybrid-bm25-mmr',
    knowledgeDir: RAG_KNOWLEDGE_DIR,
    files: state.files,
    chunks: state.chunks.length,
    avgChunkTokens: Number(state.avgTokens.toFixed(2)),
    topK: RAG_TOP_K,
    minScore: RAG_MIN_SCORE,
    mmrLambda: RAG_MMR_LAMBDA,
    lastError: state.lastError,
    loadedAt: state.loadedAt ? new Date(state.loadedAt).toISOString() : null,
  }
}

