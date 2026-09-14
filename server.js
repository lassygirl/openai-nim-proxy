// server.js - OpenAI to NVIDIA NIM API Proxy (Secure Edition)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '200mb' }));
app.use(express.urlencoded({ limit: '200mb', extended: true }));

// --- CONFIG ---
const NIM_API_BASE  = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY   = process.env.NIM_API_KEY;
const PROXY_API_KEY = process.env.PROXY_API_KEY || null;

const SHOW_REASONING       = true;
const ENABLE_THINKING_MODE = true;
const TIMEOUT_MS = 600000; // 10 minutes

// --- MODEL MAPPING ---
const MODEL_MAPPING = {
  // 'gpt-4o': deprecated (deepseek-v4-pro-0813), no replacement yet
  'gpt-4-turbo':           'deepseek-ai/deepseek-v4-flash-0731',
  // 'gpt-4': deprecated (glm-5.2 free endpoint pulled), no replacement yet
  'gpt-4-32k':             'minimaxai/minimax-m2.7',
  'gpt-4-vision':          'minimaxai/minimax-m3',
  'gemini-pro':            'moonshotai/kimi-k3',
  'gpt-3.5-turbo':         'moonshotai/kimi-k2.5',
  'gpt-3.5-turbo-instruct':'moonshotai/kimi-k2-thinking',
  'claude-3-opus':         'deepseek-ai/deepseek-v3.2',
  'claude-3-sonnet':       'google/gemma-4-31b-it',
  'claude-3-haiku':        'qwen/qwen3-coder-480b-a35b-instruct',
  'claude-instant':        'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4o-mini':           'qwen/qwen3-235b-a22b',
  'gpt-4-1106-preview':    'deepseek-ai/deepseek-v3.1',
};

// --- CONTEXT LIMITS ---
const MODEL_CONTEXT = {
  'deepseek-ai/deepseek-v4-flash-0731':         1000000,
  'deepseek-ai/deepseek-v3.2':                   128000,
  'deepseek-ai/deepseek-v3.1':                   128000,
  'minimaxai/minimax-m2.7':                       32000,
  'minimaxai/minimax-m3':                       1000000,
  'moonshotai/kimi-k3':                         1000000,
  'moonshotai/kimi-k2.5':                        128000,
  'moonshotai/kimi-k2-thinking':                 256000,
  'qwen/qwen3-coder-480b-a35b-instruct':          32000,
  'qwen/qwen3-235b-a22b':                         32000,
  'nvidia/nemotron-3-super-120b-a12b':          1000000,
  'google/gemma-4-31b-it':                       256000,
};

// --- THINKING PARAMS ---
// location 'root' = merged into request root. 'ctk' = nested under chat_template_kwargs.
const THINKING_PARAM_BUILDERS = {
  deepseek_v3: () => ({ location: 'ctk',  params: { thinking: true } }),
  deepseek_v4: () => ({ location: 'ctk',  params: { thinking: true, reasoning_effort: 'high' } }),
  kimi_k3:     () => ({ location: 'root', params: { reasoning_effort: 'max' } }),
  nemotron:    () => ({ location: 'ctk',  params: { enable_thinking: true } }),
  minimax:     () => ({ location: 'ctk',  params: { thinking_mode: 'enabled' } }),
  glm:         () => ({ location: 'ctk',  params: { enable_thinking: true } }),
};

function getModelFamily(nimModel) {
  if (nimModel === 'moonshotai/kimi-k3') return 'kimi_k3';
  if (nimModel === 'deepseek-ai/deepseek-v4-flash-0731') return 'deepseek_v4';
  if (nimModel.startsWith('deepseek-ai/')) return 'deepseek_v3';
  if (nimModel.startsWith('nvidia/nemotron')) return 'nemotron';
  if (nimModel.startsWith('minimaxai/')) return 'minimax';
  if (nimModel.startsWith('z-ai/')) return 'glm';
  return null;
}

const THINKING_ENABLED_MODELS = [
  'deepseek-ai/deepseek-v3.1',
  'deepseek-ai/deepseek-v3.2',
  'deepseek-ai/deepseek-v4-flash-0731',
  'moonshotai/kimi-k3',
  'nvidia/nemotron-3-super-120b-a12b',
  'minimaxai/minimax-m3',
];

function safeStringify(obj) {
  try { return JSON.stringify(obj); } catch (_) { return '[circular or unstringifiable]'; }
}

// --- AUTH ---
function checkAuth(req, res, next) {
  if (req.path === '/health') return next();
  if (!PROXY_API_KEY) return next();
  const authHeader = req.headers['authorization'];
  const provided = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;
  if (provided !== PROXY_API_KEY) {
    return res.status(401).json({ error: { message: 'Invalid or missing proxy API key.', type: 'authentication_error', code: 401 } });
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
  const models = Object.keys(MODEL_MAPPING).map(id => ({ id, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy' }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    console.log(`[REQ] model=${model} | max_tokens=${max_tokens} | stream=${stream}`);

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: { message: 'messages must be a non-empty array', type: 'invalid_request_error', code: 400 } });
    }

    const nimModel = MODEL_MAPPING[model] || null;
    if (!nimModel) {
      return res.status(503).json({ error: { message: `No NIM model mapped for '${model}'.`, type: 'invalid_request_error', code: 503 } });
    }

    // Strip <think> blocks from incoming history to stop payload bloat
    const stripThink = (content) => {
      if (typeof content === 'string') return content.replace(/<think>[\s\S]*?<\/think>\n*/g, '').trim();
      return content;
    };
    const cleanMessages = messages.map(m => ({ ...m, content: stripThink(m.content) }));

    // Token-aware trim: keep system msgs + first assistant msg, trim oldest history
    const estimateTokens = (msgs) => msgs.reduce((sum, m) => {
      const c = m.content;
      if (!c) return sum;
      if (typeof c === 'string') return sum + Math.ceil(c.length / 4);
      if (Array.isArray(c)) return sum + c.reduce((s, part) => s + Math.ceil((part.text || part.content || JSON.stringify(part)).length / 4), 0);
      return sum + Math.ceil(JSON.stringify(c).length / 4);
    }, 0);

    const protectedMsgs = [], chatHistory = [];
    let firstAssistantSeen = false;
    for (const msg of cleanMessages) {
      if (msg.role === 'system') protectedMsgs.push(msg);
      else if (msg.role === 'assistant' && !firstAssistantSeen) { protectedMsgs.push(msg); firstAssistantSeen = true; }
      else chatHistory.push(msg);
    }

    const contextLimit = MODEL_CONTEXT[nimModel] || 32000;
    let remaining = contextLimit - (max_tokens || 9024) - estimateTokens(protectedMsgs);
    const kept = [];
    for (let i = chatHistory.length - 1; i >= 0; i--) {
      const t = estimateTokens([chatHistory[i]]);
      if (remaining - t < 0) break;
      kept.unshift(chatHistory[i]);
      remaining -= t;
    }
    const trimmedMessages = [...protectedMsgs, ...kept];
    console.log(`[CTX] ${nimModel} | kept ${trimmedMessages.length}/${messages.length} msgs`);

    const nimRequest = {
      model: nimModel,
      messages: trimmedMessages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 9024,
      stream: stream ?? false,
    };

    if (ENABLE_THINKING_MODE && THINKING_ENABLED_MODELS.includes(nimModel)) {
      const family = getModelFamily(nimModel);
      if (family && THINKING_PARAM_BUILDERS[family]) {
        const { location, params } = THINKING_PARAM_BUILDERS[family]();
        if (location === 'root') Object.assign(nimRequest, params);
        else nimRequest.chat_template_kwargs = params; // root-level key, NOT wrapped in extra_body
      }
    }

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        Accept: stream ? 'text/event-stream' : 'application/json',
      },
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      responseType: stream ? 'stream' : 'json',
      timeout: TIMEOUT_MS
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const streamTimeout = setTimeout(() => {
        console.error(`[STREAM] Timeout after ${TIMEOUT_MS / 60000} min`);
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
              const reasoning = delta.reasoning_content ?? delta.reasoning; // NIM isn't consistent - some models use one name, some the other
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
              delete delta.reasoning;
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (_) { res.write(line + '\n'); }
        }
      });
      response.data.on('end', () => { clearTimeout(streamTimeout); if (!res.writableEnded) res.end(); });
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
          const reasoning = c.message?.reasoning_content ?? c.message?.reasoning;
          if (SHOW_REASONING && reasoning) {
            content = `<think>\n${reasoning}\n</think>\n\n${content}`;
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
        nimError.on('end', () => { try { resolve(JSON.parse(raw)); } catch { resolve(raw); } });
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