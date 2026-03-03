// server.js - OpenAI to NVIDIA NIM API Proxy (Secure Edition)

const express = require('express');

const cors = require('cors');

const axios = require('axios');



const app = express();

const PORT = process.env.PORT || 3000;



// Middleware

app.use(cors());

app.use(express.json({ limit: '50mb' }));

app.use(express.urlencoded({ limit: '50mb', extended: true }));



// ─── CONFIG ────────────────────────────────────────────────────────────────

const NIM_API_BASE  = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY   = process.env.NIM_API_KEY;

const PROXY_API_KEY = process.env.PROXY_API_KEY || null;



const SHOW_REASONING      = true;

const ENABLE_THINKING_MODE = true;



// ─── MODEL MAPPING ─────────────────────────────────────────────────────────

const MODEL_MAPPING = {

  'gpt-4o':          'deepseek-ai/deepseek-v3.2',

  'gpt-4-turbo':     'moonshotai/kimi-k2.5',

  'gpt-4':           'z-ai/glm5',

  'gpt-4-32k':       'nvidia/llama-3.1-nemotron-ultra-253b-v1',

  'gemini-pro':      'moonshotai/kimi-k2-thinking',

  'gpt-4-vision':    'z-ai/glm4.7',

  'gpt-3.5-turbo':   'qwen/qwen3-coder-480b-a35b-instruct',

  'claude-3-opus':   'moonshotai/kimi-k2-instruct-0905',

  'claude-3-sonnet': 'deepseek-ai/deepseek-v3_1',

  'claude-3-haiku':  'openai/gpt-oss-20b',

  'claude-instant':  'meta/llama-3.1-8b-instruct',

};



// ─── PER-MODEL CONTEXT LIMITS ──────────────────────────────────────────────

const MODEL_CONTEXT = {

  'z-ai/glm5':                                    120000,

  'z-ai/glm4.7':                                   32000,

  'deepseek-ai/deepseek-v3.2':                    128000,

  'deepseek-ai/deepseek-v3_1':                     64000,

  'moonshotai/kimi-k2.5':                         128000,

  'moonshotai/kimi-k2-instruct-0905':             128000,

  'moonshotai/kimi-k2-thinking':                  128000,

  'nvidia/llama-3.1-nemotron-ultra-253b-v1':       32000,

  'qwen/qwen3-coder-480b-a35b-instruct':           32000,

  'openai/gpt-oss-20b':                            32000,

  'meta/llama-3.1-8b-instruct':                    32000,

};



// ─── AUTH MIDDLEWARE ───────────────────────────────────────────────────────

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



// ─── ROUTES ────────────────────────────────────────────────────────────────

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



app.get('/v1/models', (req, res) => {

  const models = Object.keys(MODEL_MAPPING).map(id => ({

    id, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'

  }));

  res.json({ object: 'list', data: models });

});



app.post('/v1/chat/completions', async (req, res) => {

  try {

    const { model, messages, temperature, max_tokens, stream } = req.body;



    // ── Resolve NIM model (single declaration, no duplicates) ──────────────

    const nimModel = MODEL_MAPPING[model] || (() => {

      const m = model.toLowerCase();

      if (m.includes('gpt-4') || m.includes('opus') || m.includes('405b')) return 'deepseek-ai/deepseek-v3.2';

      if (m.includes('claude') || m.includes('gemini') || m.includes('70b')) return 'z-ai/glm5';

      return 'meta/llama-3.1-8b-instruct';

    })();



    // ── Smart token-aware trimming ─────────────────────────────────────────

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

    for (const msg of messages) {

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

    let trimmedMessages = [...protectedMsgs, ...kept];
    if (kept.length > 50) trimmedMessages.splice(protectedMsgs.length, kept.length - 50);

    // Trim until actual JSON payload sent to NIM is under 480KB
    const MAX_BYTES = 480 * 1024;
    while (trimmedMessages.length > protectedMsgs.length + 2) {
      const size = Buffer.byteLength(JSON.stringify(trimmedMessages), 'utf8');
      if (size <= MAX_BYTES) break;
      trimmedMessages.splice(protectedMsgs.length, 1);
    }



    console.log(`[CTX] ${nimModel} | kept ${trimmedMessages.length}/${messages.length} msgs | trimmed ${messages.length - trimmedMessages.length} oldest`);



    // ── Build and send NIM request ─────────────────────────────────────────

    const nimRequest = {

      model: nimModel,

      messages: trimmedMessages,

      temperature: temperature ?? 0.6,

      max_tokens: max_tokens ?? 9024,

      stream: stream ?? false,

      ...(ENABLE_THINKING_MODE && { extra_body: { chat_template_kwargs: { thinking: true } } })

    };



    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {

      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },

      responseType: stream ? 'stream' : 'json'

    });



    if (stream) {

      res.setHeader('Content-Type', 'text/event-stream');

      res.setHeader('Cache-Control', 'no-cache');

      res.setHeader('Connection', 'keep-alive');



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



      response.data.on('end', () => res.end());

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

    const nimError = err.response?.data;

    console.error('Proxy error:', err.message);

    console.error('NIM error:', JSON.stringify(nimError));

    res.status(err.response?.status || 500).json({

      error: {

        message: nimError?.detail || nimError?.message || err.message || 'Internal server error',

        nim_response: nimError,

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

  console.log(`\n🚀 OpenAI → NVIDIA NIM Proxy running on port ${PORT}`);

  console.log(`🔒 Proxy auth:       ${PROXY_API_KEY ? 'ENABLED' : 'DISABLED ⚠️'}`);

  console.log(`💡 Reasoning:        ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);

  console.log(`🧠 Thinking mode:    ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);

  console.log(`📋 Models mapped:    ${Object.keys(MODEL_MAPPING).length}\n`);

});
