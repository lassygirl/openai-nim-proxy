// server.js - OpenAI to NVIDIA NIM API Proxy (Secure Edition)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware - 200mb to handle large Janitor AI payloads with reasoning history
app.use(cors());
app.use(express.json({ limit: '200mb' }));
app.use(express.urlencoded({ limit: '200mb', extended: true }));

// --- CONFIG ---
const NIM_API_BASE  = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY   = process.env.NIM_API_KEY;
const PROXY_API_KEY = process.env.PROXY_API_KEY || null;

const SHOW_REASONING       = true;
const ENABLE_THINKING_MODE = true;

// --- TIMEOUT CONFIG ---
// Change this one value to adjust ALL timeouts across the proxy.
// 5 min = 300000 | 8 min = 480000 | 10 min = 600000 | 15 min = 900000
const TIMEOUT_MS = 600000; // 10 minutes

// --- MODEL MAPPING ---
const MODEL_MAPPING = {
  'gpt-4o':                'deepseek-ai/deepseek-v4-pro',         // DeepSeek V4 Pro - 1.6T params, 1M ctx, Think High/Max modes
  'gpt-4-turbo':           'deepseek-ai/deepseek-v4-flash',        // DeepSeek V4 Flash - fast version, 1M ctx
  'gpt-4':                 'z-ai/glm-5.1',                         // GLM-5.1 - 754B, 131K ctx, best open weights
  'gpt-4-32k':             'minimaxai/minimax-m2.7',               // MiniMax M2.7 - 230B, coding + reasoning
  'gpt-4-vision':          'z-ai/glm4.7',                          // GLM-4.7 - 32K ctx, your tested favourite
  'gemini-pro':            'moonshotai/kimi-k2.6',                 // Kimi K2.6 - 1T params, 32B active, multimodal
  'gpt-3.5-turbo':         'moonshotai/kimi-k2.5',                 // Kimi K2.5 - 128K ctx
  'gpt-3.5-turbo-instruct':'moonshotai/kimi-k2-thinking',          // Kimi K2 Thinking - 256K ctx, reasoning traces
  'claude-3-opus':         'deepseek-ai/deepseek-v3.2',            // DeepSeek V3.2 - 128K ctx, strong logic
  'claude-3-sonnet':       'google/gemma-4-31b-it',            // DeepSeek V3.1 - 128K ctx
  'claude-3-haiku':        'qwen/qwen3-coder-480b-a35b-instruct',  // Qwen3 Coder 480B - best coding logic
  'claude-instant':        'nvidia/nemotron-3-super-120b-a12b',    // Nemotron Super - 1M ctx, never forgets
  'gpt-4o-mini':           'qwen/qwen3-235b-a22b',                 // Qwen3 235B MoE - strong reasoning
  'gpt-4-1106-preview':    'deepseek-ai/deepseek-v3.1-terminus',   // V3.1 Terminus - hybrid think/no-think
  
};

// --- PER-MODEL CONTEXT LIMITS ---
const MODEL_CONTEXT = {
  'deepseek-ai/deepseek-v4-pro':                1000000,
  'deepseek-ai/deepseek-v4-flash':              1000000,
  'deepseek-ai/deepseek-v3.2':                  128000,
  'deepseek-ai/deepseek-v3.1':                  128000,
  'deepseek-ai/deepseek-v3.1-terminus':         128000,
  'z-ai/glm-5.1':                               131072,
  'z-ai/glm4.7':                                 32000,
  'minimaxai/minimax-m2.7':                      32000,
  'moonshotai/kimi-k2.6':                       131072,
  'moonshotai/kimi-k2.5':                       128000,
  'moonshotai/kimi-k2-thinking':                256000,
  'qwen/qwen3-coder-480b-a35b-instruct':         32000,
  'qwen/qwen3-235b-a22b':                        32000,
  'nvidia/nemotron-3-super-120b-a12b':         1000000,
};

// --- THINKING MODE SUPPORT ---
// Only models listed here receive the extra_body thinking parameter.
// Sending it to unsupported models causes instant 400 errors.
const THINKING_SUPPORTED = [
  'z-ai/glm-5.1',
  'moonshotai/kimi-k2-thinking',
  'deepseek-ai/deepseek-v3.1',
  'deepseek-ai/deepseek-v3.2',
  'deepseek-ai/deepseek-v3.1-terminus',
  'deepseek-ai/deepseek-v4-pro',
  'deepseek-ai/deepseek-v4-flash',
];

// --- SAFE JSON STRINGIFY ---
// Prevents circular reference crashes when logging network errors
function safeStringify(obj) {
  try { return JSON.stringify(obj); } catch (_) { return '[circular or unstringifiable]'; }
}

// --- AUTH MIDDLEWARE ---
function checkAuth(req, res, next) {
  if (req.path === '/health') return next();
  if (!PROXY_API_KEY) return next();
  const authHeader = req.headers['authorization'];
  const provided = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;
  if (provided !== PROXY_API_KEY) {
    return res.status(401).json({
      error: { message: 'Invalid or missing proxy API key.', type: 'authentication_error', code: 401 }
    });
  }
  next();
}
app.use(checkAuth);

// --- ROUTES ---
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    proxy_auth: PROXY_API_KEY ? 'enabled' : 'disabled',
    nim_key_set: !!NIM_API_KEY,
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    timeout_minutes: TIMEOUT_MS / 60000,
    models: Object.keys(MODEL_MAPPING).length
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    console.log(`[REQ] model=${model} | max_tokens=${max_tokens} | stream=${stream}`);

    // Validate request
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: { message: 'messages must be a non-empty array', type: 'invalid_request_error', code: 400 }
      });
    }

    // Resolve NIM model
    const nimModel = MODEL_MAPPING[model] || (() => {
      const m = model.toLowerCase();
      if (m.includes('gpt-4') || m.includes('opus') || m.includes('405b')) return 'deepseek-ai/deepseek-v4-pro';
      if (m.includes('claude') || m.includes('gemini') || m.includes('70b')) return 'z-ai/glm-5.1';
      return 'nvidia/nemotron-3-super-120b-a12b';
    })();

    // Strip <think> blocks from incoming history
    // SHOW_REASONING=true injects <think> blocks into responses which Janitor AI
    // stores and sends back - stripping prevents payload from ballooning
    const stripThink = (content) => {
      if (typeof content === 'string')
        return content.replace(/<think>[\s\S]*?<\/think>\n*/g, '').trim();
      return content;
    };
    const cleanMessages = messages.map(m => ({ ...m, content: stripThink(m.content) }));

    // Token-aware trimming
    // Protects: ALL system messages (prompt, character card, memory summaries)
    //         + first assistant message (character intro/persona)
    // Trims:    oldest regular chat exchanges only
    const estimateTokens = (msgs) =>
      msgs.reduce((sum, m) => {
        const c = m.content;
        if (!c) return sum;
        if (typeof c === 'string') return sum + Math.ceil(c.length / 4);
        if (Array.isArray(c)) return sum + c.reduce((s, part) =>
          s + Math.ceil((part.text || part.content || JSON.stringify(part)).length / 4), 0);
        return sum + Math.ceil(JSON.stringify(c).length / 4);
      }, 0);

    const tokenBudget = (MODEL_CONTEXT[nimModel] || 32000) - (max_tokens || 9024);

    const protectedMsgs = [], chatHistory = [];
    let firstAssistantSeen = false;
    for (const msg of cleanMessages) {
      if (msg.role === 'system') {
        protectedMsgs.push(msg);
      } else if (msg.role === 'assistant' && !firstAssistantSeen) {
        protectedMsgs.push(msg);
        firstAssistantSeen = true;
      } else {
        chatHistory.push(msg);
      }
    }

    const kept = [];
    let budget = tokenBudget - estimateTokens(protectedMsgs);
    for (let i = chatHistory.length - 1; i >= 0; i--) {
      const t = estimateTokens([chatHistory[i]]);
      if (budget - t < 0) break;
      kept.unshift(chatHistory[i]);
      budget -= t;
    }
    const trimmedMessages = [...protectedMsgs, ...kept];

    console.log(`[CTX] ${nimModel} | kept ${trimmedMessages.length}/${messages.length} msgs | trimmed ${messages.length - trimmedMessages.length} oldest`);

    // Build and send NIM request
    const nimRequest = {
      model: nimModel,
      messages: trimmedMessages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 9024,
      stream: stream ?? false,
      ...(ENABLE_THINKING_MODE && THINKING_SUPPORTED.includes(nimModel) && {
        extra_body: { chat_template_kwargs: { thinking: true } }
      })
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      responseType: stream ? 'stream' : 'json',
      timeout: TIMEOUT_MS
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      // Stream-level timeout - if NVIDIA hangs mid-stream, ends cleanly
      const streamTimeout = setTimeout(() => {
        console.error(`[STREAM] Timeout after ${TIMEOUT_MS / 60000} min - NVIDIA hung mid-stream, ending response`);
        if (!res.writableEnded) res.end();
      }, TIMEOUT_MS);

      let buffer = '', thinkOpen = false;

      response.data.on('data', chunk => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          if (line.includes('[DONE]')) { res.write(line + '\n'); continue; }
          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (delta) {
              const reasoning = delta.reasoning_content;
              const content   = delta.content;
              if (SHOW_REASONING) {
                let out = '';
                if (reasoning && !thinkOpen)  { out = '<think>\n' + reasoning; thinkOpen = true; }
                else if (reasoning)           { out = reasoning; }
                if (content && thinkOpen)     { out += '</think>\n\n' + content; thinkOpen = false; }
                else if (content)             { out += content; }
                if (out) delta.content = out;
              } else {
                delta.content = content ?? '';
              }
              delete delta.reasoning_content;
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (_) { res.write(line + '\n'); }
        }
      });

      response.data.on('end', () => {
        clearTimeout(streamTimeout);
        if (!res.writableEnded) res.end();
      });

      response.data.on('error', err => {
        clearTimeout(streamTimeout);
        console.error('Stream error:', err.message || safeStringify(err));
        if (!res.writableEnded) res.end();
      });

    } else {
      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map(c => {
          let content = c.message?.content ?? '';
          if (SHOW_REASONING && c.message?.reasoning_content) {
            content = `<think>\n${c.message.reasoning_content}\n</think>\n\n${content}`;
          }
          return { index: c.index, message: { role: c.message.role, content }, finish_reason: c.finish_reason };
        }),
        usage: response.data.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }

  } catch (err) {
    let nimError = err.response?.data;
    if (nimError && typeof nimError.pipe === 'function') {
      nimError = await new Promise((resolve) => {
        let raw = '';
        nimError.on('data', chunk => raw += chunk.toString());
        nimError.on('end', () => {
          try { resolve(JSON.parse(raw)); }
          catch { resolve(raw); }
        });
        nimError.on('error', () => resolve('[stream read error]'));
      });
    }
    console.error('Proxy error:', err.message);
    console.error('NIM error:', safeStringify(nimError));
    if (res.headersSent) return;
    res.status(err.response?.status || 500).json({
      error: {
        message: nimError?.detail || nimError?.message || err.message || 'Internal server error',
        type: 'invalid_request_error',
        code: err.response?.status || 500
      }
    });
  }
});

// Alias routes
app.post('/chat/completions', (req, res, next) => { req.url = '/v1/chat/completions'; app.handle(req, res, next); });
app.get('/models', (req, res, next) => { req.url = '/v1/models'; app.handle(req, res, next); });

app.get('/', (req, res) => {
  res.json({ service: 'OpenAI to NVIDIA NIM Proxy', status: 'running',
    endpoints: { health: '/health', models: '/v1/models', completions: '/v1/chat/completions' } });
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log('\nOpenAI -> NVIDIA NIM Proxy running on port ' + PORT);
  console.log('Proxy auth:      ' + (PROXY_API_KEY ? 'ENABLED' : 'DISABLED'));
  console.log('NIM key:         ' + (NIM_API_KEY ? 'SET' : 'MISSING - set NIM_API_KEY in Render!'));
  console.log('Reasoning:       ' + (SHOW_REASONING ? 'ENABLED' : 'DISABLED'));
  console.log('Thinking mode:   ' + (ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'));
  console.log('Timeout:         ' + (TIMEOUT_MS / 60000) + ' minutes');
  console.log('Models mapped:   ' + Object.keys(MODEL_MAPPING).length + '\n');
});
