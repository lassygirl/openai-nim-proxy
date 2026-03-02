// server.js - OpenAI to NVIDIA NIM API Proxy (Secure Edition)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// ─── CONFIG ────────────────────────────────────────────────────────────────

// NVIDIA NIM settings (set these in Render environment variables)
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

// 🔒 Proxy password — set PROXY_API_KEY in Render env vars, use it in Janitor AI
// If not set, the proxy is open (not recommended)
const PROXY_API_KEY = process.env.PROXY_API_KEY || null;

// 🔥 REASONING DISPLAY TOGGLE — set true to show <think> reasoning blocks
const SHOW_REASONING = true;

// 🔥 THINKING MODE TOGGLE — set true for models that support thinking parameter
const ENABLE_THINKING_MODE = false;

// ─── MODEL MAPPING ─────────────────────────────────────────────────────────
// Maps OpenAI model names → NVIDIA NIM model IDs
// Janitor AI will send one of the keys; your proxy uses the value to call NIM.

const MODEL_MAPPING = {
  // ── Latest / Flagship ──────────────────────────────────────────────────
  'gpt-4o':          'deepseek-ai/deepseek-v3.2',          // DeepSeek V3.2 685B
  'gpt-4-turbo':     'moonshotai/kimi-k2-5',               // Kimi K2.5 multimodal
  'gpt-4':           'z-ai/glm5',                          // GLM-5 744B

  // ── Strong Reasoning ───────────────────────────────────────────────────
  'gpt-4-32k':       'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gemini-pro':      'moonshotai/kimi-k2-thinking',         // Kimi K2 thinking mode

  // ── Coding / Dev ───────────────────────────────────────────────────────
  'gpt-3.5-turbo':   'qwen/qwen3-coder-480b-a35b-instruct',

  // ── Balanced ───────────────────────────────────────────────────────────
  'claude-3-opus':   'moonshotai/kimi-k2-instruct-0905',
  'claude-3-sonnet': 'deepseek-ai/deepseek-v3_1',          // V3.1 (underscore!)

  // ── Fast / Light ───────────────────────────────────────────────────────
  'claude-3-haiku':  'openai/gpt-oss-20b',
  'claude-instant':  'meta/llama-3.1-8b-instruct',
};

// ─── MIDDLEWARE: AUTH ──────────────────────────────────────────────────────

function checkAuth(req, res, next) {
  // Skip auth for health check
  if (req.path === '/health') return next();

  // If no proxy key is set, allow all (open proxy — set one in Render!)
  if (!PROXY_API_KEY) return next();

  const authHeader = req.headers['authorization'];
  const provided = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;

  if (provided !== PROXY_API_KEY) {
    return res.status(401).json({
      error: {
        message: 'Invalid or missing proxy API key.',
        type: 'authentication_error',
        code: 401
      }
    });
  }
  next();
}

app.use(checkAuth);

// ─── ROUTES ────────────────────────────────────────────────────────────────

// Health check (no auth required)
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    proxy_auth: PROXY_API_KEY ? 'enabled' : 'disabled ⚠️',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    models: Object.keys(MODEL_MAPPING).length
  });
});

// List available models
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Main proxy: Chat Completions
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Smart model resolution with fallback chain
    let nimModel = MODEL_MAPPING[model];

    if (!nimModel) {
      // Try passing the model name directly to NIM first
      try {
        const test = await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: s => s < 500
        });
        if (test.status >= 200 && test.status < 300) nimModel = model;
      } catch (_) {}
    }

    if (!nimModel) {
      // Fallback by keyword
      const m = model.toLowerCase();
      if (m.includes('gpt-4') || m.includes('opus') || m.includes('405b')) {
        nimModel = 'deepseek-ai/deepseek-v3_2';
      } else if (m.includes('claude') || m.includes('gemini') || m.includes('70b')) {
        nimModel = 'zai-org/GLM-5';
      } else {
        nimModel = 'meta/llama-3.1-8b-instruct';
      }
    }

    // Build NIM request
    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 9024,
      stream: stream ?? false,
      ...(ENABLE_THINKING_MODE && { extra_body: { chat_template_kwargs: { thinking: true } } })
    };

    // Forward to NVIDIA NIM
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let thinkOpen = false;

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

      response.data.on('end',   ()    => res.end());
      response.data.on('error', err => { console.error('Stream error:', err); res.end(); });

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
    console.error('Proxy error:', err.message);
    res.status(err.response?.status || 500).json({
      error: { message: err.message || 'Internal server error', type: 'invalid_request_error', code: err.response?.status || 500 }
    });
  }
});

// Alias routes — handle Janitor AI calling without /v1 prefix
app.post('/chat/completions', (req, res, next) => {
  req.url = '/v1/chat/completions';
  app.handle(req, res, next);
});

app.get('/models', (req, res, next) => {
  req.url = '/v1/models';
  app.handle(req, res, next);
});

// Root — show friendly info instead of 404
app.get('/', (req, res) => {
  res.json({
    service: 'OpenAI to NVIDIA NIM Proxy',
    status: 'running',
    endpoints: {
      health:      '/health',
      models:      '/v1/models',
      completions: '/v1/chat/completions'
    }
  });
});

// 404 catch-all
app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`\n🚀 OpenAI → NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`🔒 Proxy auth:       ${PROXY_API_KEY ? 'ENABLED' : 'DISABLED ⚠️'}`);
  console.log(`💡 Reasoning:        ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`🧠 Thinking mode:    ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`📋 Models mapped:    ${Object.keys(MODEL_MAPPING).length}\n`);
});
