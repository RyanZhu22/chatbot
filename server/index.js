import 'dotenv/config'
import express from 'express'
import { getRagStatus, retrieveKnowledge } from './rag.js'

const app = express()
const PORT = process.env.PORT || 3001
const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS || 300000)
const STREAM_HEARTBEAT_MS = Number(process.env.STREAM_HEARTBEAT_MS || 15000)
const DEFAULT_PROVIDER = (process.env.LLM_PROVIDER || 'openai').toLowerCase()
const SUPPORTED_PROVIDERS = new Set(['openai', 'deepseek'])
const SYSTEM_PROMPT =
  process.env.SYSTEM_PROMPT ||
  'You are an internal company assistant. Give concise and accurate answers. Format responses in Markdown.'

app.use(express.json({ limit: '1mb' }))

function normalizeProvider(provider) {
  const target = String(provider || DEFAULT_PROVIDER).toLowerCase()
  if (!SUPPORTED_PROVIDERS.has(target)) {
    return null
  }
  return target
}

function getProviderConfig(provider) {
  const targetProvider = normalizeProvider(provider)
  if (!targetProvider) {
    return null
  }

  if (targetProvider === 'deepseek') {
    const maxTokens = Number(process.env.DEEPSEEK_MAX_TOKENS || 0)
    return {
      provider: 'deepseek',
      apiKey: process.env.DEEPSEEK_API_KEY || process.env.OPENAI_API_KEY,
      baseUrl: (process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com').replace(/\/+$/, ''),
      model: process.env.DEEPSEEK_MODEL || 'deepseek-chat',
      maxTokens: Number.isFinite(maxTokens) && maxTokens > 0 ? maxTokens : undefined,
    }
  }

  const maxTokens = Number(process.env.OPENAI_MAX_TOKENS || 0)
  return {
    provider: 'openai',
    apiKey: process.env.OPENAI_API_KEY,
    baseUrl: (process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/+$/, ''),
    model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
    maxTokens: Number.isFinite(maxTokens) && maxTokens > 0 ? maxTokens : undefined,
  }
}

function buildMessages(userMessage, history, contextMessage) {
  const safeHistory = Array.isArray(history)
    ? history
        .filter((item) => item && (item.role === 'user' || item.role === 'assistant'))
        .map((item) => ({
          role: item.role,
          content: String(item.content || '').trim(),
        }))
        .filter((item) => item.content.length > 0)
        .slice(-20)
    : []

  const systemMessages = [{ role: 'system', content: SYSTEM_PROMPT }]
  if (contextMessage) {
    systemMessages.push({
      role: 'system',
      content: contextMessage,
    })
  }

  return [...systemMessages, ...safeHistory, { role: 'user', content: userMessage }]
}

async function requestModelStream(messages, provider, onDelta, signal) {
  const config = getProviderConfig(provider)
  if (!config) {
    throw new Error('Unsupported provider. Use openai or deepseek.')
  }

  if (!config.apiKey) {
    throw new Error(
      config.provider === 'deepseek'
        ? 'DEEPSEEK_API_KEY is missing'
        : 'OPENAI_API_KEY is missing',
    )
  }

  let response
  try {
    response = await fetch(`${config.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${config.apiKey}`,
      },
      body: JSON.stringify({
        model: config.model,
        temperature: 0.2,
        stream: true,
        ...(config.maxTokens ? { max_tokens: config.maxTokens } : {}),
        messages,
      }),
      signal,
    })
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('REQUEST_ABORTED')
    }
    const code =
      error instanceof Error && 'cause' in error && error.cause && typeof error.cause === 'object'
        ? error.cause.code
        : undefined
    const causeMessage =
      error instanceof Error && 'cause' in error && error.cause && typeof error.cause === 'object'
        ? error.cause.message
        : undefined
    const extra = [code ? `code=${code}` : '', causeMessage ? `cause=${causeMessage}` : '']
      .filter(Boolean)
      .join(' ')
    throw new Error(
      `Upstream fetch failed (${config.provider} ${config.baseUrl})${extra ? ` ${extra}` : ''}`,
    )
  }

  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    const errorMessage = data?.error?.message || `Model request failed with status ${response.status}`
    throw new Error(errorMessage)
  }

  if (!response.body) {
    throw new Error('Model did not return a readable stream')
  }

  const decoder = new TextDecoder()
  let buffer = ''
  let finishReason = null
  let sawDoneSentinel = false
  let shouldStop = false

  const processLine = (line) => {
    const trimmed = line.trim()
    if (!trimmed || !trimmed.startsWith('data:')) {
      return false
    }
    const payload = trimmed.slice(5).trim()
    if (!payload || payload === '[DONE]') {
      if (payload === '[DONE]') {
        sawDoneSentinel = true
      }
      return payload === '[DONE]'
    }

    let parsed
    try {
      parsed = JSON.parse(payload)
    } catch {
      return false
    }

    const choice = parsed?.choices?.[0]
    if (typeof choice?.finish_reason === 'string' && choice.finish_reason.length > 0) {
      finishReason = choice.finish_reason
    }

    const delta = choice?.delta?.content
    if (typeof delta === 'string' && delta.length > 0) {
      onDelta(delta)
    }
    return false
  }

  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true })
    let lineBreakIndex = buffer.indexOf('\n')
    while (lineBreakIndex !== -1) {
      const line = buffer.slice(0, lineBreakIndex)
      buffer = buffer.slice(lineBreakIndex + 1)
      const isDone = processLine(line)
      if (isDone) {
        shouldStop = true
        break
      }
      lineBreakIndex = buffer.indexOf('\n')
    }
    if (shouldStop) {
      break
    }
  }

  if (buffer && !shouldStop) {
    processLine(buffer)
  }

  return {
    finishReason,
    sawDoneSentinel,
  }
}

app.get('/api/health', async (_req, res) => {
  const config = getProviderConfig()
  if (!config) {
    return res.status(500).json({ ok: false, error: 'Invalid default LLM_PROVIDER' })
  }
  const rag = await getRagStatus()
  res.json({
    ok: true,
    service: 'chat-api',
    provider: config.provider,
    model: config.model,
    hasApiKey: Boolean(config.apiKey),
    baseUrl: config.baseUrl,
    rag,
  })
})

app.post('/api/chat', async (req, res) => {
  const userMessage = String(req.body?.message || '').trim()
  const history = req.body?.history
  const provider = req.body?.provider

  if (!userMessage) {
    return res.status(400).json({ error: 'message is required' })
  }

  if (provider && !normalizeProvider(provider)) {
    return res.status(400).json({
      error: 'invalid_provider',
      detail: 'provider must be openai or deepseek',
    })
  }

  let timeoutId
  let heartbeatId
  let didTimeout = false
  let clientDisconnected = false
  let fullReply = ''

  try {
    const ragResult = await retrieveKnowledge(userMessage)
    const messages = buildMessages(userMessage, history, ragResult.contextMessage)
    const config = getProviderConfig(provider)
    const controller = new AbortController()
    timeoutId = setTimeout(() => {
      didTimeout = true
      controller.abort()
    }, REQUEST_TIMEOUT_MS)

    req.on('aborted', () => {
      clientDisconnected = true
      controller.abort()
    })

    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8')
    res.setHeader('Cache-Control', 'no-cache, no-transform')
    res.setHeader('Connection', 'keep-alive')
    res.flushHeaders?.()
    heartbeatId = setInterval(() => {
      if (!res.writableEnded) {
        res.write(': ping\n\n')
      }
    }, STREAM_HEARTBEAT_MS)

    res.write(
      `event: meta\ndata: ${JSON.stringify({
        provider: config?.provider,
        model: config?.model,
      })}\n\n`,
    )
    res.write(
      `event: context\ndata: ${JSON.stringify({
        citations: ragResult.citations,
        matchedChunks: ragResult.matches.length,
      })}\n\n`,
    )

    const streamResult = await requestModelStream(
      messages,
      provider,
      (delta) => {
        fullReply += delta
        res.write(`event: delta\ndata: ${JSON.stringify({ text: delta })}\n\n`)
      },
      controller.signal,
    )

    if (!fullReply.trim()) {
      throw new Error('Model returned an empty response')
    }

    const finishReason =
      streamResult && typeof streamResult.finishReason === 'string'
        ? streamResult.finishReason
        : null

    const donePayload = {
      timestamp: new Date().toISOString(),
      finishReason,
    }

    if (!streamResult?.sawDoneSentinel) {
      donePayload.partial = true
      donePayload.detail = 'stream_closed_unexpectedly'
    } else if (finishReason && finishReason !== 'stop') {
      donePayload.partial = true
      donePayload.detail = `finish_reason_${finishReason}`
    }

    res.write(`event: done\ndata: ${JSON.stringify(donePayload)}\n\n`)
    return res.end()
  } catch (error) {
    const detail = error instanceof Error ? error.message : 'Unknown server error'

    if (detail === 'REQUEST_ABORTED') {
      if (clientDisconnected) {
        return res.end()
      }

      const timeoutDetail = didTimeout
        ? `Model request timed out after ${Math.ceil(REQUEST_TIMEOUT_MS / 1000)}s`
        : 'Request was aborted'

      if (fullReply.trim()) {
        if (!res.writableEnded) {
          res.write(
            `event: done\ndata: ${JSON.stringify({
              timestamp: new Date().toISOString(),
              partial: true,
              detail: timeoutDetail,
            })}\n\n`,
          )
        }
        return res.end()
      }

      if (!res.headersSent) {
        return res.status(504).json({
          error: 'chat_request_failed',
          detail: timeoutDetail,
        })
      }

      res.write(`event: error\ndata: ${JSON.stringify({ detail: timeoutDetail })}\n\n`)
      return res.end()
    }

    if (!res.headersSent) {
      return res.status(500).json({
        error: 'chat_request_failed',
        detail,
      })
    }
    res.write(
      `event: error\ndata: ${JSON.stringify({
        detail,
      })}\n\n`,
    )
    return res.end()
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    if (heartbeatId) {
      clearInterval(heartbeatId)
    }
  }
})

app.listen(PORT, () => {
  console.log(`Chat API listening on http://localhost:${PORT}`)
})
