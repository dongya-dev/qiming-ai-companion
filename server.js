const express = require('express');
const cors = require('cors');
const path = require('path');
const crypto = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ---------- config ----------
const DEEPSEEK_KEY = process.env.DEEPSEEK_API_KEY || '';
const DEEPSEEK_URL = 'https://api.deepseek.com/v1/chat/completions';

// ---------- middleware ----------
app.use(cors());
app.use(express.json({ limit: '5mb' }));
// 访问根路径自动跳转到登录页
app.get('/', (req, res) => res.redirect('/login.html'));

// ---------- helpers ----------

/** Call DeepSeek chat API with retry */
async function deepseek(messages, temperature = 0.5, maxTokens = 3000, retries = 1) {
  if (!DEEPSEEK_KEY) {
    const err = new Error('DeepSeek API Key 未配置，请在环境变量中设置 DEEPSEEK_API_KEY');
    err.status = 503;
    throw err;
  }
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 25000);

      const resp = await fetch(DEEPSEEK_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${DEEPSEEK_KEY}`
        },
        body: JSON.stringify({
          model: 'deepseek-chat',
          messages,
          temperature,
          max_tokens: maxTokens
        }),
        signal: controller.signal
      });

      clearTimeout(timeout);

      if (!resp.ok) {
        const body = await resp.text();
        const err = new Error(`DeepSeek API ${resp.status}: ${body.slice(0, 300)}`);
        err.status = resp.status;
        throw err;
      }

      const data = await resp.json();
      return data.choices[0].message.content;
    } catch (err) {
      lastError = err;
      if (attempt < retries) {
        console.warn(`DeepSeek API retry ${attempt + 1}/${retries}: ${err.message}`);
        await new Promise(r => setTimeout(r, 1000));
      }
    }
  }
  throw lastError;
}

/** Robust JSON extraction from AI text (handles markdown fences) */
function extractJSON(raw) {
  try { return JSON.parse(raw); } catch (_) { /* continue */ }

  let m = raw.match(/```json\s*([\s\S]*?)```/);
  if (m) { try { return JSON.parse(m[1].trim()); } catch (_) {} }

  m = raw.match(/```\s*([\s\S]*?)```/);
  if (m) { try { return JSON.parse(m[1].trim()); } catch (_) {} }

  m = raw.match(/\{[\s\S]*\}/);
  if (m) {
    // Use bracket-depth counting for robust extraction (handles nested JSON in code blocks)
    const start = m.index;
    let depth = 0, end = start;
    for (let i = start; i < raw.length; i++) {
      if (raw[i] === '{') depth++;
      else if (raw[i] === '}') { depth--; if (depth === 0) { end = i + 1; break; } }
    }
    const jsonCandidate = raw.slice(start, end);
    try { return JSON.parse(jsonCandidate); } catch (_) {}
    // Fallback: try the original regex match
    try { return JSON.parse(m[0]); } catch (_) {}
  }

  return null;
}

/** Local fallback mindmap: simple keyword frequency extraction */
function generateLocalMindmap(text) {
  // 按句号/换行分段，提取每段首句关键词
  const segments = text.split(/[。！？!?\n；;]+/).filter(s => s.trim().length > 0);
  if (segments.length === 0) {
    return { center: '文本要点', branches: [{ name: '内容', color: '#FF7B42', children: ['请查看简化文本'] }] };
  }

  // 简单词频统计（排除常见停用词）
  const stopWords = new Set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']);
  const wordFreq = {};
  const allWords = text.replace(/[^\u4e00-\u9fff]/g, ' ').split(/\s+/).filter(w => w.length >= 2 && !stopWords.has(w));
  allWords.forEach(w => { wordFreq[w] = (wordFreq[w] || 0) + 1; });

  // 取高频词作为分支
  const sorted = Object.entries(wordFreq).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const colors = ['#FF7B42', '#9B5DE5', '#06D6A0', '#118AB2', '#EF476F', '#FFB347'];

  if (sorted.length === 0) {
    // 用段落首词
    const branchNames = segments.slice(0, 6).map(s => s.slice(0, 6));
    return {
      center: '文本要点',
      branches: branchNames.map((name, i) => ({
        name, color: colors[i], children: [segments[i] ? segments[i].slice(0, 8) + '…' : '查看详情']
      }))
    };
  }

  const branches = sorted.slice(0, 5).map(([word, freq], i) => {
    // 找到包含该词的句子作为子节点
    const relatedSegments = segments.filter(s => s.includes(word)).slice(0, 3);
    const children = relatedSegments.map(s => s.length > 8 ? s.slice(0, 8) + '…' : s);
    return {
      name: word,
      color: colors[i % colors.length],
      children: children.length > 0 ? children : [`相关: ${word}`]
    };
  });

  return {
    center: text.length > 10 ? text.slice(0, 8) + '…' : text,
    branches
  };
}

// ---------- 豆包 TTS (火山引擎语音合成 - HTTP Chunked) ----------
const DOUBAO_APP_ID = process.env.DOUBAO_APP_ID || '';
const DOUBAO_ACCESS_KEY = process.env.DOUBAO_ACCESS_KEY || '';
const DOUBAO_RESOURCE_ID = 'seed-tts-2.0';
const DOUBAO_TTS_URL = 'https://openspeech.bytedance.com/api/v3/tts/unidirectional';

/**
 * 调用豆包 TTS HTTP Chunked 单向流式 API 合成语音
 * @param {string} text - 要合成的文本
 * @param {string} speaker - 发音人
 * @returns {Promise<Buffer>} mp3 音频数据
 */
async function doubaoTTS(text, speaker = 'zh_female_xiaohe_uranus_bigtts') {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30000);

  try {
    console.log(`🔊 TTS 请求: url=${DOUBAO_TTS_URL}, appId=${DOUBAO_APP_ID.slice(0,4)}..., resourceId=${DOUBAO_RESOURCE_ID}`);
    const body = JSON.stringify({
      user: { uid: 'qiming_user' },
      req_params: {
        text,
        speaker,
        audio_params: {
          format: 'mp3',
          sample_rate: 24000
        }
      }
    });
    console.log(`🔊 TTS body: ${body.slice(0, 150)}...`);

    const resp = await fetch(DOUBAO_TTS_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Api-App-Id': DOUBAO_APP_ID,
        'X-Api-Access-Key': DOUBAO_ACCESS_KEY,
        'X-Api-Resource-Id': DOUBAO_RESOURCE_ID
      },
      body,
      signal: controller.signal
    });

    console.log(`📡 豆包 HTTP ${resp.status}, headers: ${[...resp.headers.entries()].map(([k,v]) => `${k}:${v}`).join('; ')}`);

    if (!resp.ok) {
      const errBody = await resp.text();
      throw new Error(`豆包 TTS HTTP ${resp.status}: ${errBody.slice(0, 200)}`);
    }

    // 获取原始响应文本。Node.js 22+ 的 resp.text() 已内置 Brotli 解压，
    // 但某些边缘情况（chunked + br 组合）可能抛异常，需要手动解压回退。
    let rawText;
    try {
      rawText = await resp.text();
    } catch (txtErr) {
      console.warn('⚠ resp.text() 失败，启用 Brotli 手动解压:', txtErr.message);
      const buf = new Uint8Array(await resp.arrayBuffer());
      try {
        // Brotli 解压（Node 11.7+ 内置支持）
        rawText = new TextDecoder('utf-8').decode(
          require('zlib').brotliDecompressSync(buf)
        );
        console.log('✅ Brotli 手动解压成功');
      } catch (brErr) {
        // 不是 Brotli 压缩，当作纯 UTF-8 文本
        console.warn('⚠ Brotli 解压失败，当纯文本处理:', brErr.message);
        rawText = new TextDecoder('utf-8').decode(buf);
      }
    }
    console.log(`📡 豆包响应长度: ${rawText.length} 字节`);
    console.log(`📡 豆包响应前200字: ${rawText.slice(0, 200)}`);

    const audioBufs = [];
    let chunkCount = 0;

    // 策略1: 按换行分割（正常情况）
    let lines = rawText.split('\n').filter(l => l.trim());

    // 策略2: 如果只有一行，尝试按 JSON 对象边界分割
    if (lines.length <= 1 && rawText.includes('}{')) {
      console.log('⚠ 换行分割失败，尝试按 JSON 边界重新分割...');
      lines = [];
      let depth = 0, start = 0;
      for (let i = 0; i < rawText.length; i++) {
        if (rawText[i] === '{') depth++;
        else if (rawText[i] === '}') {
          depth--;
          if (depth === 0) {
            lines.push(rawText.slice(start, i + 1));
            start = i + 1;
          }
        }
      }
      console.log(`📡 JSON边界分割: ${lines.length} 个片段`);
    }

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      chunkCount++;
      try {
        const packet = JSON.parse(trimmed);
        if (packet.code === 20000000) {
          console.log(`🏁 豆包合成完成`, packet.usage ? `字数:${packet.usage.text_words}` : '');
        } else if (packet.code === 0 && packet.data) {
          audioBufs.push(Buffer.from(packet.data, 'base64'));
        } else if (packet.code !== 0) {
          console.warn('⚠ 豆包 TTS chunk 异常:', JSON.stringify(packet).slice(0, 200));
        }
      } catch (e) {
        console.warn('⚠ 豆包行解析失败:', trimmed.slice(0, 100), e.message);
      }
    }

    console.log(`📊 豆包 TTS 统计: ${chunkCount} 个chunk, ${audioBufs.length} 个音频块`);

    if (audioBufs.length === 0) {
      // 打印前3条非空行供调试
      const preview = rawText.split('\n').filter(l => l.trim()).slice(0, 3).join('\n');
      throw new Error(`豆包 TTS 未返回音频数据 (${lines.length}行, ${chunkCount}chunk)\n响应前3行: ${preview.slice(0, 300)}`);
    }

    return Buffer.concat(audioBufs);
  } finally {
    clearTimeout(timer);
  }
}

// ---------- API: 语音合成 (豆包 TTS) ----------
app.post('/api/speech', async (req, res) => {
  try {
    const { text, speaker } = req.body;
    if (!text || text.trim().length === 0) {
      return res.status(400).json({ error: '请提供要朗读的文本' });
    }
    if (text.length > 500) {
      return res.status(400).json({ error: '文本过长，单次最多500字' });
    }
    if (!DOUBAO_APP_ID || !DOUBAO_ACCESS_KEY) {
      return res.status(503).json({ error: '语音合成服务未配置（缺少豆包API凭证），请联系管理员设置 DOUBAO_APP_ID 和 DOUBAO_ACCESS_KEY' });
    }

    console.log(`🔊 TTS 合成: "${text.slice(0, 40)}${text.length > 40 ? '…' : ''}" (${text.length}字)`);
    const audioBuf = await doubaoTTS(text.trim(), speaker);
    console.log(`✅ TTS 完成: ${(audioBuf.length / 1024).toFixed(1)} KB`);

    res.set({
      'Content-Type': 'audio/mpeg',
      'Content-Length': audioBuf.length,
      'Cache-Control': 'no-cache'
    });
    res.end(audioBuf);
  } catch (err) {
    console.error('TTS error:', err.message);
    res.status(500).json({ error: '语音合成失败', detail: err.message });
  }
});

// ---------- API: 阅读 - 文本简化 ----------
app.post('/api/reading/simplify', async (req, res) => {
  try {
    const { text, difficulty } = req.body;
    if (!text || text.trim().length < 10) {
      return res.status(400).json({ error: '文本太短，请输入至少10个字的内容' });
    }

    // Build difficulty-specific instructions
    let difficultyPrompt = '';
    if (difficulty === 'basic') {
      difficultyPrompt = `【简化级别：基础（大幅简化）】
- 大幅度缩短句子，每句8-18字
- 用最简单的日常词汇替换所有术语和生僻词
- 每个专业概念后加括号用大白话解释
- 确保小学三年级以上就能读懂
- 保留核心意思即可，大胆删减修饰语和不重要的细节
- 为每句生成拼音标注（带声调，空格分隔）`;
    } else if (difficulty === 'pinyin') {
      difficultyPrompt = `【模式：仅拼音标注】
- 原文完全保留不变，不简化任何内容
- 只做长句自然断句，每句15-30字
- 为每句生成准确的拼音标注（带声调）
- 多音字务必根据上下文选择正确读音`;
    } else {
      difficultyPrompt = `【简化级别：标准】
- 长句拆分为短句，每句10-25字
- 用简单词汇替换生僻词和复杂术语
- 保持核心意思
- 为每句生成拼音标注（带声调，空格分隔）`;
    }

    const result = await deepseek([
      {
        role: 'system',
        content: `你是一个专门帮助阅读障碍学生的教育AI助手。

【理论基础】
- 简单阅读观(Gough & Tunmer, 1986)：阅读=解码×语言理解
- 语音缺陷假说(Stanovich, 1988)：拼音提供显式语音线索
- 认知负荷理论(Sweller, 1988)：降低内在认知负荷
- 双编码理论(Paivio, 1971)：拼音+文字促进双通道加工

${difficultyPrompt}

只输出JSON：
{"sentences": [{"simplified": "简化短句", "pinyin": "jiǎn huà duǎn jù"}, ...]}

注意：simplified纯中文，pinyin为对应拼音，多音字根据上下文选择正确读音。`
      },
      { role: 'user', content: `请简化以下文本：\n\n${text}` }
    ], 0.3);

    const data = extractJSON(result);

    if (data && data.sentences && Array.isArray(data.sentences)) {
      res.json(data);
    } else {
      const sentences = text
        .split(/[。！？!?\n；;]+/)
        .map(s => s.trim())
        .filter(s => s.length > 0)
        .map(s => ({ simplified: s, pinyin: '' }));
      res.json({ sentences, note: 'AI返回格式异常，已使用基础拆分' });
    }
  } catch (err) {
    console.error('Simplify error:', err.message);
    res.status(500).json({ error: 'AI服务暂时不可用，请稍后重试', detail: err.message });
  }
});

// ---------- API: 阅读 - 思维导图 ----------
app.post('/api/reading/mindmap', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || text.trim().length < 10) {
      return res.status(400).json({ error: '文本太短' });
    }

    let aiGenerated = false;
    let data = null;

    try {
      const result = await deepseek([
        {
          role: 'system',
          content: `你是教育AI助手。根据文本生成知识思维导图。

【理论基础】
- PASS智力理论：思维导图激活同时性加工
- 双编码理论：图像化组织+文字双重编码
- 认知负荷理论：层级结构降低认知负荷

要求：
1. 提取核心主题为中心节点
2. 生成3-6个主要分支
3. 每分支1-3个子节点
4. 分支颜色从以下选：#FF7B42, #9B5DE5, #06D6A0, #118AB2, #EF476F, #FFB347

只输出JSON：
{"center": "中心主题", "branches": [{"name": "分支", "color": "#FF7B42", "children": ["子节点"]}]}

分支名和子节点2-6字为宜。`
        },
        { role: 'user', content: `为以下内容生成思维导图：\n\n${text}` }
      ], 0.5, 1500, 0);

      data = extractJSON(result);
      if (data && data.center) {
        aiGenerated = true;
      }
    } catch (apiErr) {
      console.warn('Mindmap API failed, using local fallback:', apiErr.message);
    }

    // Fallback to local generation
    if (!data || !data.center) {
      data = generateLocalMindmap(text);
    }

    res.json({
      ...data,
      _source: aiGenerated ? 'ai' : 'local',
      _note: aiGenerated ? '' : '（本地生成，AI服务暂不可用）'
    });
  } catch (err) {
    console.error('Mindmap error:', err.message);
    // Ultimate fallback
    res.json({
      center: '文本要点',
      branches: [
        { name: '核心内容', color: '#FF7B42', children: ['请查看简化文本'] },
        { name: '关键概念', color: '#9B5DE5', children: ['AI本地生成'] }
      ],
      _source: 'fallback',
      _note: '（思维导图生成受限，请检查网络连接）'
    });
  }
});

// ---------- API: 任务拆解 ----------
app.post('/api/task/break', async (req, res) => {
  try {
    const { task, difficulty, goal } = req.body;
    if (!task || task.trim().length < 4) {
      return res.status(400).json({ error: '任务描述太短' });
    }

    const diffLabel = difficulty === 'hard' ? '困难' : difficulty === 'medium' ? '中等' : '简单';
    const diffInstruction = difficulty === 'hard'
      ? '拆解步骤稍大但仍具体，鼓励语有挑战性'
      : difficulty === 'medium'
      ? '保持适中步骤粒度'
      : '拆得更细(8-10步)，每步更小更易启动，鼓励语更温暖';

    const goalInstruction = goal
      ? `用户设定了目标："${goal}"。请参考此目标调整步骤。`
      : '';

    const result = await deepseek([
      {
        role: 'system',
        content: `你是帮助ADHD学生的AI学习伴侣。

【理论基础】
- 执行功能理论(Barkley, 1997)：外部化步骤补偿工作记忆困难
- 心流理论：挑战与技能匹配
- 自我决定理论：能力感+自主感+归属感
- 自我调节学习：拆解=计划→勾选=执行→反思=反思

要求：
1. 分解为6-10个微步骤（简单难度更细）
2. 每步5-10分钟内可完成
3. 描述具体可操作（说出第一个物理动作）
4. 每步附温暖鼓励语（5-15字）
5. 前几步要特别简单建立信心
6. ${diffInstruction}
7. ${goalInstruction}

只输出JSON：{"steps": [{"step": "具体步骤", "encouragement": "鼓励语"}]}
step 8-20字，encouragement 5-15字。`
      },
      { role: 'user', content: `请分解任务：${task}` }
    ], 0.6);

    const data = extractJSON(result);
    res.json(data || {
      steps: [
        { step: '先深呼吸，准备好学习用品', encouragement: '开始就是胜利！' },
        { step: '仔细阅读任务要求', encouragement: '慢慢来，不着急' },
        { step: '把任务分成几小块', encouragement: '你已经很厉害了！' },
        { step: '完成第一小块', encouragement: '第一步完成，真棒！' },
        { step: '继续完成剩下的部分', encouragement: '保持节奏，你可以的！' },
        { step: '检查并完善', encouragement: '快完成了，加油！' },
        { step: '整体回顾和确认', encouragement: '你做到了！' }
      ]
    });
  } catch (err) {
    console.error('Break task error:', err.message);
    res.status(500).json({ error: '任务拆解失败', detail: err.message });
  }
});

// ==================== NEW: API - AI 智能答疑 ====================
app.post('/api/tutor/chat', async (req, res) => {
  try {
    const { message, context } = req.body;
    if (!message || message.trim().length < 2) {
      return res.status(400).json({ error: '请输入你的问题' });
    }

    const contextBlock = context
      ? `【学习上下文：当前正在阅读以下内容】\n${context.slice(0, 800)}\n【回答问题时请结合此上下文】`
      : '';

    const result = await deepseek([
      {
        role: 'system',
        content: `你是启明AI学习伴侣中的智能答疑老师。你的学生可能有阅读障碍或ADHD。

【理论基础】
- 最近发展区(ZPD)：在学生的当前水平上稍作延伸
- 认知负荷理论：分解复杂概念，每次聚焦一个要点
- 多感官学习：用比喻、类比、实例帮助理解
- 自我决定理论：温暖鼓励，增强学习动机

回答要求：
1. 用简单清晰的语言解释（适合中小学生阅读水平）
2. 尽量举一个生活中的例子帮助理解
3. 如果概念复杂，分2-3个小点逐步解释
4. 结尾加一句鼓励语
5. 回答控制在100-300字之间

${contextBlock}

你的回答直接就是纯文本，不需要JSON格式。`
      },
      { role: 'user', content: message }
    ], 0.7, 800);

    res.json({ reply: result.trim() });
  } catch (err) {
    console.error('Tutor error:', err.message);
    res.status(500).json({ error: '答疑服务暂不可用', detail: err.message });
  }
});

// ==================== NEW: API - 知识闪卡生成 ====================
app.post('/api/flashcard/generate', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || text.trim().length < 20) {
      return res.status(400).json({ error: '文本太短，请输入至少20字的内容' });
    }

    const result = await deepseek([
      {
        role: 'system',
        content: `你是一个教育AI助手。根据文本内容生成知识闪卡（Flashcards）。

【理论基础】
- 测试效应(Roediger & Karpicke, 2006)：提取练习比重复阅读更有效
- 间隔重复(Ebbinghaus, 1885)：分散复习对抗遗忘曲线
- 精细加工理论：正反面促进深度加工而非表面记忆

要求：
1. 从文本中提取5-10个关键知识点
2. 每张闪卡：正面=问题/术语/概念，反面=答案/解释/定义
3. 正面简洁（2-10字），反面清晰（10-40字）
4. 由易到难排列

只输出JSON：
{"cards": [{"front": "术语或问题", "back": "解释或答案"}, ...]}`
      },
      { role: 'user', content: `从以下内容生成闪卡：\n\n${text.slice(0, 2000)}` }
    ], 0.5, 1500);

    const data = extractJSON(result);
    res.json(data || {
      cards: [
        { front: '这段文字的主题是什么？', back: text.slice(0, 40) + '…' },
        { front: '关键概念1', back: '请仔细阅读原文找出答案' }
      ]
    });
  } catch (err) {
    console.error('Flashcard error:', err.message);
    // Local fallback
    const segments = text.split(/[。！？!?\n]+/).filter(s => s.trim().length > 8);
    const cards = segments.slice(0, 5).map((s, i) => ({
      front: `要点 ${i + 1}`,
      back: s.trim().length > 50 ? s.trim().slice(0, 50) + '…' : s.trim()
    }));
    res.json({ cards: cards.length > 0 ? cards : [{ front: '请尝试输入更多内容', back: '闪卡需要较丰富的文本才能生成' }] });
  }
});

// ==================== NEW: API - 学习笔记结构化 ====================
app.post('/api/notes/summarize', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || text.trim().length < 20) {
      return res.status(400).json({ error: '笔记太短，请输入至少20字的内容' });
    }

    const result = await deepseek([
      {
        role: 'system',
        content: `你是学习笔记整理助手。将原始笔记整理为结构化学习笔记。

【理论基础】
- 生成效应(Slamecka & Graf, 1978)：主动组织比被动阅读记忆更深
- 精细加工理论：结构化重组促进深度语义加工
- 元认知策略：提纲帮助建立知识框架，监控理解

要求：
1. 提取标题（概括内容，5-10字）
2. 生成3-6个要点的提纲（每个10-25字）
3. 提取3-5个关键术语及其定义
4. 生成一个30字以内的一句话总结

只输出JSON：
{"title": "标题", "outline": ["要点1", "要点2", ...], "keyTerms": [{"term": "术语", "definition": "定义"}, ...], "summary": "一句话总结"}`
      },
      { role: 'user', content: `请整理以下笔记：\n\n${text.slice(0, 2500)}` }
    ], 0.5, 1500);

    const data = extractJSON(result);
    res.json(data || {
      title: '学习笔记',
      outline: text.split(/[。！？!?\n]+/).filter(s => s.trim()).slice(0, 5).map(s => s.trim().slice(0, 30)),
      keyTerms: [],
      summary: text.slice(0, 30) + '…'
    });
  } catch (err) {
    console.error('Notes error:', err.message);
    res.status(500).json({ error: '笔记整理失败', detail: err.message });
  }
});

// ==================== 飞书多维表格云端存储 ====================
const FEISHU = {
  APP_ID: process.env.FEISHU_APP_ID || '',
  APP_SECRET: process.env.FEISHU_APP_SECRET || '',
  BASE_TOKEN: process.env.FEISHU_BASE_TOKEN || '',
  USER_ID: 'default',
  TABLES: {
    achievements: 'tblxlGB3DRj1BoYK',
    history: 'tblGHr7W3Hm8w0yX',
    flashcard_decks: 'tbl6P6cxZtnWEmff',
    pomodoro_log: 'tblC0noVXYBm48jn',
    users: 'tblExlYOx0VzdddG'
  },
  // 用户表字段名（飞书 API 使用 field_name，非 field_id）
  USER_FIELDS: {
    username: 'username',
    password_hash: 'password_hash',
    role: 'role',
    created_at: 'created_at'
  }
};

/** 飞书服务是否已配置（三项凭证缺一不可） */
function feishuConfigured() {
  return !!(FEISHU.APP_ID && FEISHU.APP_SECRET && FEISHU.BASE_TOKEN);
}

// ── 认证密钥（HMAC签名用） ──
const AUTH_SECRET = 'qiming_edu_auth_2026_secret_key';

// Token 缓存（提前5分钟刷新）
let _feishuToken = { value: null, expire: 0 };

async function feishuAuth() {
  if (_feishuToken.value && Date.now() < _feishuToken.expire) return _feishuToken.value;
  const resp = await fetch('https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ app_id: FEISHU.APP_ID, app_secret: FEISHU.APP_SECRET })
  });
  const d = await resp.json();
  if (d.code !== 0) throw new Error(`飞书认证失败(${d.code}): ${d.msg}`);
  _feishuToken.value = d.tenant_access_token;
  _feishuToken.expire = Date.now() + (d.expire - 300) * 1000;
  console.log('🔑 飞书 Token 已刷新');
  return _feishuToken.value;
}

async function feishuCall(method, path, body = null) {
  const token = await feishuAuth();
  const opts = { method, headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(`https://open.feishu.cn/open-apis${path}`, opts);
  const d = await resp.json();
  if (d.code !== 0) {
    console.error(`[飞书] ${method} ${path} → ${d.code}: ${d.msg}`);
    throw new Error(d.msg || `code=${d.code}`);
  }
  return d.data;
}

// ==================== 用户认证系统 ====================

// ── 密码哈希：PBKDF2 + 随机盐 ──
function hashPassword(password, salt = null) {
  if (!salt) salt = crypto.randomBytes(16).toString('hex');
  const hash = crypto.pbkdf2Sync(password, salt, 10000, 64, 'sha512').toString('hex');
  return `${salt}:${hash}`;
}

function verifyPassword(password, stored) {
  const [salt] = stored.split(':');
  return hashPassword(password, salt) === stored;
}

// ── 简易令牌：Base64(username:timestamp:HMAC) ──
function createToken(username) {
  const ts = Date.now();
  const payload = `${username}:${ts}`;
  const sig = crypto.createHmac('sha256', AUTH_SECRET).update(payload).digest('hex');
  return Buffer.from(`${username}:${ts}:${sig}`).toString('base64');
}

function verifyToken(token) {
  try {
    const decoded = Buffer.from(token, 'base64').toString('utf8');
    const parts = decoded.split(':');
    if (parts.length !== 3) return null;
    const [username, ts, sig] = parts;
    // 令牌有效期7天
    if (Date.now() - parseInt(ts) > 7 * 24 * 3600 * 1000) return null;
    const expected = crypto.createHmac('sha256', AUTH_SECRET).update(`${username}:${ts}`).digest('hex');
    if (sig !== expected) return null;
    return username;
  } catch (e) { return null; }
}

// ── 用户表 CRUD ──
async function fsFindUser(username) {
  const records = await fsListRecords('users');
  const F = FEISHU.USER_FIELDS;
  return records.find(r => r.fields[F.username] === username) || null;
}

async function fsCreateUser(username, passwordHash, role) {
  const F = FEISHU.USER_FIELDS;
  return await fsBatchCreate('users', [{
    [F.username]: username,
    [F.password_hash]: passwordHash,
    [F.role]: role,
    [F.created_at]: new Date().toISOString()
  }]);
}

// ── 注册 ──
app.post('/api/auth/register', async (req, res) => {
  try {
    if (!feishuConfigured()) {
      return res.status(503).json({ error: '用户注册功能未配置（缺少飞书多维表格凭证）。请使用游客模式，或联系管理员设置 FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_BASE_TOKEN' });
    }
    const { username, password, role } = req.body;
    if (!username || !password) return res.status(400).json({ error: '用户名和密码不能为空' });
    if (username.length < 3 || username.length > 12) return res.status(400).json({ error: '用户名需3-12个字符' });
    if (password.length < 4) return res.status(400).json({ error: '密码至少4位' });
    if (!/^[\w\u4e00-\u9fff]{3,12}$/.test(username)) return res.status(400).json({ error: '用户名只能包含中文、字母、数字或下划线' });

    const ex = await fsFindUser(username);
    if (ex) return res.status(409).json({ error: '该用户名已被注册' });

    const hashed = hashPassword(password);
    await fsCreateUser(username, hashed, role || '学生');
    const token = createToken(username);

    console.log(`✅ 新用户注册: ${username} (${role || '学生'})`);
    res.json({ ok: true, username, role: role || '学生', token });
  } catch (err) {
    console.error('[Auth/Register]', err.message);
    res.status(500).json({ error: '注册失败，请稍后重试' });
  }
});

// ── 登录 ──
app.post('/api/auth/login', async (req, res) => {
  try {
    if (!feishuConfigured()) {
      return res.status(503).json({ error: '用户登录功能未配置（缺少飞书多维表格凭证）。请使用游客模式，或联系管理员设置 FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_BASE_TOKEN' });
    }
    const { username, password } = req.body;
    if (!username || !password) return res.status(400).json({ error: '请填写用户名和密码' });

    const user = await fsFindUser(username);
    if (!user) return res.status(401).json({ error: '用户不存在' });

    const F = FEISHU.USER_FIELDS;
    if (!verifyPassword(password, user.fields[F.password_hash])) {
      return res.status(401).json({ error: '密码错误' });
    }

    const role = user.fields[F.role] || '学生';
    const token = createToken(username);

    console.log(`🔑 用户登录: ${username} (${role})`);
    res.json({ ok: true, username, role, token });
  } catch (err) {
    console.error('[Auth/Login]', err.message);
    res.status(500).json({ error: '登录失败，请稍后重试' });
  }
});

// ── 验证令牌 ──
app.post('/api/auth/verify', async (req, res) => {
  if (!feishuConfigured()) {
    return res.status(503).json({ error: '用户认证服务未配置' });
  }
  const { token } = req.body;
  if (!token) return res.status(400).json({ error: '缺少认证令牌' });

  const username = verifyToken(token);
  if (!username) return res.status(401).json({ error: '令牌无效或已过期' });

  try {
    const user = await fsFindUser(username);
    if (!user) return res.status(401).json({ error: '用户不存在' });
    const F = FEISHU.USER_FIELDS;
    res.json({ ok: true, username, role: user.fields[F.role] || '学生' });
  } catch (err) {
    res.status(500).json({ error: '验证失败' });
  }
});

// ── 游客模式（服务器端生成游客昵称） ──
app.post('/api/auth/guest', (_req, res) => {
  const names = ['探索者','好奇星人','快乐学习家','小书虫','智慧芽','萌新','晨光','追梦人','小太阳','极光'];
  const name = names[Math.floor(Math.random() * names.length)] + Math.floor(Math.random() * 100);
  const token = createToken(name);
  res.json({ ok: true, username: name, role: '学生', isGuest: true, token });
});

// ── 通用 CRUD ──
async function fsListRecords(tableKey) {
  const data = await feishuCall('GET',
    `/bitable/v1/apps/${FEISHU.BASE_TOKEN}/tables/${FEISHU.TABLES[tableKey]}/records?page_size=500`);
  return data.items || [];
}

async function fsBatchCreate(tableKey, records) {
  if (records.length === 0) return;
  return await feishuCall('POST',
    `/bitable/v1/apps/${FEISHU.BASE_TOKEN}/tables/${FEISHU.TABLES[tableKey]}/records/batch_create`,
    { records: records.map(fields => ({ fields })) });
}

async function fsBatchDelete(tableKey, recordIds) {
  if (recordIds.length === 0) return;
  return await feishuCall('POST',
    `/bitable/v1/apps/${FEISHU.BASE_TOKEN}/tables/${FEISHU.TABLES[tableKey]}/records/batch_delete`,
    { records: recordIds });
}

async function fsUpdateRecord(tableKey, recordId, fields) {
  return await feishuCall('PUT',
    `/bitable/v1/apps/${FEISHU.BASE_TOKEN}/tables/${FEISHU.TABLES[tableKey]}/records/${recordId}`,
    { fields });
}

// ── achievements: upsert 单条（按 userId 隔离） ──
async function cloudSaveAchievements(userId, data) {
  const records = await fsListRecords('achievements');
  const ex = records.find(r => r.fields.user_id === userId);
  const fields = {
    user_id: userId,
    total: data.total || 0,
    streak: data.streak || 0,
    words: data.words || 0,
    tasks: data.tasks || 0,
    focusMinutes: data.focusMinutes || 0,
    lastDate: data.lastDate || ''
  };
  if (ex) { await fsUpdateRecord('achievements', ex.record_id, fields); }
  else { await fsBatchCreate('achievements', [fields]); }
}

async function cloudLoadAchievements(userId) {
  const records = await fsListRecords('achievements');
  const ex = records.find(r => r.fields.user_id === userId);
  if (ex) {
    const f = ex.fields;
    return { total: Number(f.total) || 0, streak: Number(f.streak) || 0, words: Number(f.words) || 0, tasks: Number(f.tasks) || 0, focusMinutes: Number(f.focusMinutes) || 0, lastDate: f.lastDate || '' };
  }
  return null;
}

// ── history: 按用户全量替换 ──
async function cloudSaveHistory(userId, data) {
  const records = await fsListRecords('history');
  const userRecords = records.filter(r => r.fields.user_id === userId);
  if (userRecords.length > 0) {
    await fsBatchDelete('history', userRecords.map(r => r.record_id));
  }
  if (data.length > 0) {
    const batch = data.map(item => ({
      user_id: userId,
      type: item.type || '',
      data: String(item.data || '').slice(0, 5000),
      time: item.time || ''
    }));
    for (let i = 0; i < batch.length; i += 500) {
      await fsBatchCreate('history', batch.slice(i, i + 500));
    }
  }
}

async function cloudLoadHistory(userId) {
  const records = await fsListRecords('history');
  return records
    .filter(r => r.fields.user_id === userId)
    .map(r => ({
      type: r.fields.type || '',
      data: r.fields.data || '',
      time: r.fields.time || ''
    }));
}

// ── flashcard_decks: 单条 JSON payload（按 userId 隔离） ──
async function cloudSaveFlashcardDecks(userId, data) {
  const records = await fsListRecords('flashcard_decks');
  const ex = records.find(r => r.fields.user_id === userId);
  const fields = { user_id: userId, deck_data: JSON.stringify(data) };
  if (ex) { await fsUpdateRecord('flashcard_decks', ex.record_id, fields); }
  else { await fsBatchCreate('flashcard_decks', [fields]); }
}

async function cloudLoadFlashcardDecks(userId) {
  const records = await fsListRecords('flashcard_decks');
  const ex = records.find(r => r.fields.user_id === userId);
  if (ex && ex.fields.deck_data) {
    try { return JSON.parse(ex.fields.deck_data); } catch (e) { return null; }
  }
  return null;
}

// ── pomodoro_log: 按用户全量替换 ──
async function cloudSavePomodoroLog(userId, data) {
  const records = await fsListRecords('pomodoro_log');
  const userRecords = records.filter(r => r.fields.user_id === userId);
  if (userRecords.length > 0) {
    await fsBatchDelete('pomodoro_log', userRecords.map(r => r.record_id));
  }
  if (data.length > 0) {
    const batch = data.map(item => ({
      user_id: userId,
      duration: Number(item.minutes) || 0,
      goal: String(item.goal || ''),
      date: item.date || ''
    }));
    for (let i = 0; i < batch.length; i += 500) {
      await fsBatchCreate('pomodoro_log', batch.slice(i, i + 500));
    }
  }
}

async function cloudLoadPomodoroLog(userId) {
  const records = await fsListRecords('pomodoro_log');
  return records
    .filter(r => r.fields.user_id === userId)
    .map(r => ({
      minutes: Number(r.fields.duration) || 0,
      goal: r.fields.goal || '',
      date: r.fields.date || ''
    }));
}

// ── 从请求中提取当前用户 ──
function extractUserId(req) {
  const auth = req.headers.authorization || '';
  if (auth.startsWith('Bearer ')) {
    return verifyToken(auth.slice(7)) || FEISHU.USER_ID;
  }
  // 兼容：从请求体或查询参数中获取
  return req.body.userId || req.query.userId || FEISHU.USER_ID;
}

// ── 处理器映射（惰性工厂：接受 userId 返回处理器） ──
function getHandlers(userId) {
  return {
    qm_ach:            { save: d => cloudSaveAchievements(userId, d), load: () => cloudLoadAchievements(userId) },
    qm_history:         { save: d => cloudSaveHistory(userId, d),      load: () => cloudLoadHistory(userId) },
    qm_flashcard_decks: { save: d => cloudSaveFlashcardDecks(userId, d), load: () => cloudLoadFlashcardDecks(userId) },
    qm_pomodoro_log:    { save: d => cloudSavePomodoroLog(userId, d),  load: () => cloudLoadPomodoroLog(userId) }
  };
}

// POST /api/sync/save  — 保存单键数据（按当前用户）
app.post('/api/sync/save', async (req, res) => {
  try {
    if (!feishuConfigured()) {
      return res.status(503).json({ error: '云同步未配置（缺少飞书凭证）' });
    }
    const userId = extractUserId(req);
    const { key, data } = req.body;
    const handlers = getHandlers(userId);
    if (!handlers[key]) return res.status(400).json({ error: `Unknown key: ${key}` });
    await handlers[key].save(data);
    console.log(`☁ 已保存: ${key} (user:${userId})`);
    res.json({ ok: true, key, userId });
  } catch (err) {
    console.error(`[Sync Save] ${req.body.key}:`, err.message);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/sync/load  — 加载单键数据（按当前用户）
app.post('/api/sync/load', async (req, res) => {
  try {
    if (!feishuConfigured()) {
      return res.status(503).json({ error: '云同步未配置（缺少飞书凭证）' });
    }
    const userId = extractUserId(req);
    const { key } = req.body;
    const handlers = getHandlers(userId);
    if (!handlers[key]) return res.status(400).json({ error: `Unknown key: ${key}` });
    const data = await handlers[key].load();
    res.json({ ok: true, key, userId, data });
  } catch (err) {
    console.error(`[Sync Load] ${req.body.key}:`, err.message);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/sync/load-all  — 一次性加载全部数据（按当前用户）
app.post('/api/sync/load-all', async (req, res) => {
  try {
    if (!feishuConfigured()) {
      return res.status(503).json({ error: '云同步未配置（缺少飞书凭证）' });
    }
    const userId = extractUserId(req);
    const handlers = getHandlers(userId);
    const result = {};
    for (const [key, handler] of Object.entries(handlers)) {
      try { result[key] = await handler.load(); }
      catch (e) { result[key] = null; console.warn(`[Sync LoadAll] ${key}:`, e.message); }
    }
    res.json({ ok: true, userId, data: result });
  } catch (err) {
    console.error(`[Sync LoadAll]:`, err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------- health check ----------
app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    model: DEEPSEEK_KEY ? 'deepseek-chat' : 'unavailable',
    tts: (DOUBAO_APP_ID && DOUBAO_ACCESS_KEY) ? 'doubao-tts' : 'unavailable',
    sync: feishuConfigured() ? 'feishu-bitable' : 'unavailable',
    auth: feishuConfigured() ? 'feishu-bitable' : 'guest-only',
    endpoints: ['simplify', 'mindmap', 'task-break', 'tutor-chat', 'flashcard', 'notes', 'speech', 'sync', 'auth'],
    timestamp: new Date().toISOString()
  });
});

// ---------- static files ----------
app.use(express.static(__dirname));

// ---------- start ----------
app.listen(PORT, () => {
  console.log('🦉 启明 AI学习伴侣 服务已启动');
  console.log(`   本地访问: http://localhost:${PORT}`);
  if (DEEPSEEK_KEY) {
    console.log(`   🤖 DeepSeek AI: 已配置 ✅`);
  } else {
    console.warn(`   ⚠ DeepSeek AI: 未配置 ❌ (设置 DEEPSEEK_API_KEY 后重启)`);
  }
  if (DOUBAO_APP_ID && DOUBAO_ACCESS_KEY) {
    console.log(`   🔊 豆包 TTS: 已配置 ✅`);
  } else {
    console.warn(`   ⚠ 豆包 TTS: 未配置 ⚠ (语音朗读不可用)`);
  }
  if (feishuConfigured()) {
    console.log(`   ☁ 飞书云同步: 已配置 ✅`);
  } else {
    console.warn(`   ⚠ 飞书云同步: 未配置 ⚠ (注册/登录/云端存储不可用，游客模式正常)`);
  }
  console.log(`   端点: 阅读简化 | 思维导图 | 任务拆解 | AI答疑 | 闪卡 | 笔记 | 语音合成 | 飞书云同步 | 用户认证`);
});
