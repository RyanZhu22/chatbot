import { useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

const sampleChats = ['Product FAQ Bot', 'Internal Policy Notes', 'Onboarding Assistant']

function normalizeErrorMessage(message) {
  const text = String(message || '请求失败')
  if (/finish_reason_length/i.test(text)) {
    return '达到单次输出上限，可点击重新生成继续'
  }
  if (/stream_closed_unexpectedly/i.test(text)) {
    return '流式连接意外中断，请重试'
  }
  if (/finish_reason_content_filter/i.test(text)) {
    return '输出被安全策略截断'
  }
  if (/This operation was aborted|The operation was aborted|REQUEST_ABORTED/i.test(text)) {
    return '连接中断或请求超时，请重试'
  }
  if (/timed out|timeout|timed out after/i.test(text)) {
    return '模型响应超时，请稍后重试'
  }
  return text
}

function findLastUserIndex(messageList) {
  for (let index = messageList.length - 1; index >= 0; index -= 1) {
    if (messageList[index]?.role === 'user') {
      return index
    }
  }
  return -1
}

function findLastAssistantIndex(messageList) {
  for (let index = messageList.length - 1; index >= 0; index -= 1) {
    if (messageList[index]?.role === 'assistant') {
      return index
    }
  }
  return -1
}

function App() {
  const [provider, setProvider] = useState('openai')
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: '你好，我是内部 Chatbot。现在已接入真实大模型接口。',
      citations: [],
    },
  ])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const abortControllerRef = useRef(null)
  const pauseRequestedRef = useRef(false)

  const startGeneration = async ({ requestMessage, baseHistory, preparedMessages }) => {
    setMessages(preparedMessages)
    setIsSending(true)
    pauseRequestedRef.current = false

    const requestController = new AbortController()
    abortControllerRef.current = requestController

    let receivedAnyDelta = false
    let receivedChars = 0
    let assistantTextBuffer = ''
    let reachedOutputCap = false

    const MAX_CONTINUE_RETRIES = 1
    const MAX_ASSISTANT_CHARS = 4500
    const CONTINUE_PROMPT = '请从上一次停止的位置继续输出，不要重复已输出内容。'

    const updateLastAssistant = (updater) => {
      setMessages((prev) => {
        if (prev.length === 0) {
          return prev
        }
        const lastIndex = prev.length - 1
        const lastMessage = prev[lastIndex]
        if (lastMessage.role !== 'assistant') {
          return prev
        }
        const updated = [...prev]
        updated[lastIndex] = updater(lastMessage)
        return updated
      })
    }

    const stopAssistantLoading = () => {
      updateLastAssistant((lastMessage) => ({
        ...lastMessage,
        isLoading: false,
      }))
    }

    const setAssistantCitations = (citations) => {
      const list = Array.isArray(citations)
        ? citations
            .map((item) => ({ source: String(item?.source || '').trim() }))
            .filter((item) => item.source)
        : []

      updateLastAssistant((lastMessage) => ({
        ...lastMessage,
        citations: list,
      }))
    }

    const trimOverlap = (existingText, incomingText) => {
      const source = String(existingText || '')
      const incoming = String(incomingText || '')
      if (!incoming) {
        return ''
      }

      const maxCheck = Math.min(300, source.length, incoming.length)
      for (let size = maxCheck; size >= 16; size -= 1) {
        if (source.endsWith(incoming.slice(0, size))) {
          return incoming.slice(size)
        }
      }
      return incoming
    }

    const appendAssistantText = (text) => {
      if (!text) {
        return
      }

      const merged = trimOverlap(assistantTextBuffer, text)
      if (!merged) {
        return
      }

      const remaining = MAX_ASSISTANT_CHARS - assistantTextBuffer.length
      if (remaining <= 0) {
        reachedOutputCap = true
        return
      }

      const bounded = merged.slice(0, remaining)
      if (!bounded) {
        reachedOutputCap = true
        return
      }

      if (bounded.length < merged.length) {
        reachedOutputCap = true
      }

      receivedAnyDelta = true
      receivedChars += bounded.length
      assistantTextBuffer += bounded

      updateLastAssistant((lastMessage) => ({
        ...lastMessage,
        text: `${lastMessage.text}${bounded}`,
        isLoading: false,
      }))
    }

    const streamOnce = async (message, history) => {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: requestController.signal,
        body: JSON.stringify({
          provider,
          message,
          history,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData?.detail || 'Request failed')
      }
      if (!response.body) {
        throw new Error('浏览器不支持流式响应')
      }

      const localReader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let donePayload = null
      let gotDeltaInThisRound = false

      const processEventBlock = (block) => {
        const lines = block.split('\n')
        let eventName = 'message'
        const dataLines = []

        for (const rawLine of lines) {
          const line = rawLine.trimEnd()
          if (!line) {
            continue
          }
          if (line.startsWith('event:')) {
            eventName = line.slice(6).trim()
            continue
          }
          if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trim())
          }
        }

        if (dataLines.length === 0) {
          return
        }

        const dataText = dataLines.join('\n')
        let payload = {}
        try {
          payload = JSON.parse(dataText)
        } catch {
          payload = {}
        }

        if (eventName === 'context') {
          setAssistantCitations(payload?.citations)
          return
        }
        if (eventName === 'delta') {
          gotDeltaInThisRound = true
          appendAssistantText(payload?.text || '')
          return
        }
        if (eventName === 'done') {
          donePayload = payload || {}
          stopAssistantLoading()
          return
        }
        if (eventName === 'error') {
          throw new Error(payload?.detail || '请求失败')
        }
      }

      while (true) {
        const { done, value: chunk } = await localReader.read()
        if (done) {
          break
        }

        buffer += decoder.decode(chunk, { stream: true }).replace(/\r\n/g, '\n').replace(/\r/g, '\n')
        let splitIndex = buffer.indexOf('\n\n')
        while (splitIndex !== -1) {
          const eventBlock = buffer.slice(0, splitIndex)
          buffer = buffer.slice(splitIndex + 2)
          processEventBlock(eventBlock)
          splitIndex = buffer.indexOf('\n\n')
        }
      }

      if (buffer.trim()) {
        processEventBlock(buffer)
      }

      if (donePayload) {
        return donePayload
      }
      if (gotDeltaInThisRound) {
        return { partial: true, detail: 'stream_closed_unexpectedly' }
      }
      return {}
    }

    const shouldAutoContinue = (payload) => {
      if (!payload?.partial) {
        return false
      }
      if (pauseRequestedRef.current || requestController.signal.aborted) {
        return false
      }
      if (reachedOutputCap || assistantTextBuffer.length >= MAX_ASSISTANT_CHARS) {
        return false
      }
      return payload?.detail === 'finish_reason_length'
    }

    try {
      let donePayload = await streamOnce(requestMessage, baseHistory)
      let retryCount = 0

      while (shouldAutoContinue(donePayload) && retryCount < MAX_CONTINUE_RETRIES) {
        const charsBefore = receivedChars
        retryCount += 1
        donePayload = await streamOnce(CONTINUE_PROMPT, [
          ...baseHistory,
          { role: 'user', content: requestMessage },
          { role: 'assistant', content: assistantTextBuffer },
        ])
        if (receivedChars === charsBefore) {
          break
        }
      }

      if (reachedOutputCap) {
        appendAssistantText('\n\n> 已达到输出长度上限，已停止继续生成。')
      } else if (donePayload?.partial && donePayload?.detail && receivedChars < 40) {
        appendAssistantText(`\n\n> 生成中断：${normalizeErrorMessage(donePayload.detail)}`)
      }

      stopAssistantLoading()
    } catch (error) {
      if (pauseRequestedRef.current) {
        updateLastAssistant((lastMessage) => ({
          ...lastMessage,
          isLoading: false,
          text: lastMessage.text || '已暂停生成',
        }))
        return
      }

      const detail =
        error instanceof Error
          ? normalizeErrorMessage(error.message)
          : '接口请求失败，请确认后端服务已启动并配置 OPENAI_API_KEY。'

      if (receivedAnyDelta && /连接中断或请求超时|模型响应超时/.test(detail)) {
        stopAssistantLoading()
        return
      }

      setMessages((prev) => {
        if (prev.length === 0) {
          return [{ role: 'assistant', text: `请求失败：${detail}`, citations: [] }]
        }

        const lastIndex = prev.length - 1
        const lastMessage = prev[lastIndex]
        if (lastMessage.role === 'assistant' && String(lastMessage.text || '').trim()) {
          const updated = [...prev]
          updated[lastIndex] = {
            ...lastMessage,
            isLoading: false,
            text: `${lastMessage.text}\n\n> 生成中断：${detail}`,
          }
          return updated
        }

        if (lastMessage.role === 'assistant') {
          return [
            ...prev.slice(0, -1),
            { role: 'assistant', text: `请求失败：${detail}`, citations: [] },
          ]
        }

        return [...prev, { role: 'assistant', text: `请求失败：${detail}`, citations: [] }]
      })
    } finally {
      if (abortControllerRef.current === requestController) {
        abortControllerRef.current = null
      }
      pauseRequestedRef.current = false
      setIsSending(false)
    }
  }

  const handleSend = async () => {
    const value = input.trim()
    if (!value || isSending) {
      return
    }

    setInput('')
    const baseHistory = messages.map((item) => ({
      role: item.role,
      content: item.text,
    }))

    await startGeneration({
      requestMessage: value,
      baseHistory,
      preparedMessages: [
        ...messages,
        { role: 'user', text: value },
        { role: 'assistant', text: '', isLoading: true, citations: [] },
      ],
    })
  }

  const handlePause = () => {
    if (!isSending || !abortControllerRef.current) {
      return
    }
    pauseRequestedRef.current = true
    abortControllerRef.current.abort()
  }

  const handleSendOrPause = async () => {
    if (isSending) {
      handlePause()
      return
    }
    await handleSend()
  }

  const handleRegenerate = async () => {
    if (isSending) {
      return
    }

    const lastUserIndex = findLastUserIndex(messages)
    if (lastUserIndex < 0) {
      return
    }

    const lastUserText = String(messages[lastUserIndex]?.text || '').trim()
    if (!lastUserText) {
      return
    }

    const historyBeforeLastUser = messages.slice(0, lastUserIndex).map((item) => ({
      role: item.role,
      content: item.text,
    }))

    await startGeneration({
      requestMessage: lastUserText,
      baseHistory: historyBeforeLastUser,
      preparedMessages: [
        ...messages.slice(0, lastUserIndex + 1),
        { role: 'assistant', text: '', isLoading: true, citations: [] },
      ],
    })
  }

  const canRegenerate = !isSending && findLastUserIndex(messages) >= 0
  const lastAssistantIndex = findLastAssistantIndex(messages)

  return (
    <div className="chat-app">
      <aside className="sidebar">
        <button className="new-chat-btn" type="button">
          + New chat
        </button>

        <div className="chat-list">
          {sampleChats.map((item) => (
            <button className="chat-item" key={item} type="button">
              {item}
            </button>
          ))}
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <span>Internal Chatbot</span>
          <label className="provider-select-wrap">
            <span>Provider</span>
            <select
              value={provider}
              onChange={(event) => setProvider(event.target.value)}
              disabled={isSending}
            >
              <option value="openai">OpenAI</option>
              <option value="deepseek">DeepSeek</option>
            </select>
          </label>
        </header>

        <section className="messages">
          {messages.map((message, index) => (
            <article className={`message message-${message.role}`} key={`${index}-${message.role}`}>
              <div className="avatar">{message.role === 'assistant' ? 'AI' : 'You'}</div>
              <div className="message-content">
                {message.role === 'assistant' ? (
                  message.isLoading && !message.text ? (
                    <div className="loading-indicator" aria-label="AI 正在思考" role="status">
                      <span />
                      <span />
                      <span />
                    </div>
                  ) : (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown>
                  )
                ) : (
                  <p>{message.text}</p>
                )}

                {message.role === 'assistant' &&
                Array.isArray(message.citations) &&
                message.citations.length > 0 ? (
                  <div className="citation-box">
                    <div className="citation-title">参考来源</div>
                    <ul className="citation-list">
                      {message.citations.map((item) => (
                        <li key={item.source} className="citation-item">
                          {item.source}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>

              {index === lastAssistantIndex && canRegenerate ? (
                <div className="inline-output-actions">
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={handleRegenerate}
                    disabled={!canRegenerate}
                  >
                    重新生成
                  </button>
                </div>
              ) : null}
            </article>
          ))}
        </section>

        <footer className="composer">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey && !isSending) {
                event.preventDefault()
                handleSendOrPause()
              }
            }}
            placeholder="Message Internal Chatbot"
            rows={1}
          />
          <button
            type="button"
            className={`btn-primary${isSending ? ' btn-primary-pause' : ''}`}
            onClick={handleSendOrPause}
            disabled={!isSending && !input.trim()}
          >
            {isSending ? '暂停' : '发送'}
          </button>
        </footer>
      </main>
    </div>
  )
}

export default App
