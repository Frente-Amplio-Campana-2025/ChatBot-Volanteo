// Importar Transformers.js desde CDN
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

env.allowLocalModels = false;

// Variables globales
let propuestasDatabase = [];
let embeddings = [];
let categoryIndex = {};
let extractor = null;
let isModelLoaded = false;

const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

const CACHE_KEY = 'chatbot_embeddings_cache';
const CACHE_VERSION = 'v3.0'; // Cambiar a v3 para propuestas

// ====== SISTEMA DE CATEGORIZACIÓN ======

const CATEGORY_KEYWORDS = {
    'asistencia-social': ['imas', 'sinirube', 'pobreza', 'vulnerabilidad', 'asistencia', 'beneficiarios', 'fis'],
    'pensiones': ['pensión', 'pensiones', 'jubilación', 'adulto mayor', 'cuidadoras', '65 años', '60 años'],
    'educacion': ['avancemos', 'comedor', 'escolar', 'estudiante', 'beca', 'mep', 'secundaria', 'primaria'],
    'cuido': ['cecudi', 'cuido', 'cen-cinai', 'madres', 'infantil', 'red de cuido'],
    'trabajo': ['trabajo', 'laboral', 'fodesaf', 'inspección', 'patrono', 'salario', 'mtss'],
    'poblaciones-vulnerables': ['calle', 'abandono', 'iafa', 'psc', 'discapacidad', 'vulnerable'],
    'salud': ['salud', 'ccss', 'médico', 'hospital', 'ebais', 'ministerio de salud', 'comunidad', 'personal de salud'],
    'salud-mental': ['mental', 'adicción', 'drogas', 'ansiedad', 'depresión', 'suicida', 'eisaa'],
    'prevencion-salud': ['prevención', 'promoción', 'entorno', 'saludable', 'recreativo'],
    'reformas-legales': ['ley', 'reforma', 'proyecto', 'legal', 'reglamento'],
    'migracion': ['migrante', 'dimex', 'migración', 'extranjero']
};

function detectCategories(query) {
    const queryLower = query.toLowerCase();
    const scores = {};
    
    for (const [category, keywords] of Object.entries(CATEGORY_KEYWORDS)) {
        let score = 0;
        for (const keyword of keywords) {
            if (queryLower.includes(keyword)) {
                score += 1;
            }
        }
        if (score > 0) {
            scores[category] = score;
        }
    }
    
    return Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .map(([cat]) => cat);
}

function buildCategoryIndex() {
    categoryIndex = {};
    
    propuestasDatabase.forEach((item, index) => {
        const category = item.categoria || 'general';
        if (!categoryIndex[category]) {
            categoryIndex[category] = [];
        }
        categoryIndex[category].push(index);
    });
    
    console.log('📑 Índice de categorías:', Object.keys(categoryIndex).map(
        cat => `${cat}: ${categoryIndex[cat].length}`
    ));
}

function getRelevantSubset(query) {
    const categories = detectCategories(query);
    
    if (categories.length === 0) {
        return propuestasDatabase.map((_, idx) => idx);
    }
    
    const indices = new Set();
    categories.slice(0, 3).forEach(cat => {
        if (categoryIndex[cat]) {
            categoryIndex[cat].forEach(idx => indices.add(idx));
        }
    });
    
    if (indices.size < 20 && categoryIndex['general']) {
        categoryIndex['general'].forEach(idx => indices.add(idx));
    }
    
    const result = Array.from(indices);
    console.log(`🎯 Búsqueda en ${result.length}/${propuestasDatabase.length} propuestas (${categories.join(', ')})`);
    
    return result;
}

// ====== FUNCIONES DE EMBEDDINGS ======

function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    
    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);
    
    if (normA === 0 || normB === 0) return 0;
    
    return dotProduct / (normA * normB);
}

async function getEmbedding(text) {
    if (!extractor) {
        throw new Error('El modelo no está cargado');
    }
    
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

// ====== CACHE ======

async function saveEmbeddingsToCache(embeddings, propuestasDatabase) {
    try {
        const cacheData = {
            version: CACHE_VERSION,
            timestamp: Date.now(),
            embeddings: embeddings,
            propuestas: propuestasDatabase.map(item => item.propuesta.substring(0, 100)),
            count: propuestasDatabase.length
        };
        
        localStorage.setItem(CACHE_KEY, JSON.stringify(cacheData));
        console.log('✅ Embeddings guardados en cache');
    } catch (error) {
        console.warn('⚠️ No se pudo guardar en cache:', error);
    }
}

async function loadEmbeddingsFromCache(propuestasDatabase) {
    try {
        const cached = localStorage.getItem(CACHE_KEY);
        if (!cached) return null;
        
        const cacheData = JSON.parse(cached);
        
        if (cacheData.version !== CACHE_VERSION || 
            cacheData.count !== propuestasDatabase.length) {
            localStorage.removeItem(CACHE_KEY);
            return null;
        }
        
        console.log('✅ Embeddings cargados desde cache');
        return cacheData.embeddings;
        
    } catch (error) {
        console.warn('⚠️ Error al cargar cache:', error);
        return null;
    }
}

// ====== CARGA DE MODELO ======

function showLoadingStatus(message, progress = null) {
    const welcomeMsg = chatContainer.querySelector('.welcome-message');
    if (welcomeMsg) {
        const statusDiv = welcomeMsg.querySelector('.loading-status') || document.createElement('div');
        statusDiv.className = 'loading-status';
        
        let html = `<p style="color: #666; margin-top: 20px;">⏳ ${message}</p>`;
        if (progress !== null) {
            html += `
                <div style="width: 100%; background: #f0f0f0; border-radius: 10px; overflow: hidden; margin-top: 10px;">
                    <div style="width: ${progress}%; background: linear-gradient(135deg, #f6df08 0%, #e6cf00 100%); height: 20px; transition: width 0.3s;"></div>
                </div>
                <p style="color: #999; font-size: 12px; margin-top: 5px;">${progress}%</p>
            `;
        }
        
        statusDiv.innerHTML = html;
        
        if (!statusDiv.parentNode) {
            welcomeMsg.appendChild(statusDiv);
        }
    }
}

async function loadModel() {
    try {
        showLoadingStatus('Descargando modelo de IA...', 10);
        
        extractor = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small');
        
        showLoadingStatus('Modelo cargado correctamente', 30);
        isModelLoaded = true;
        
        return true;
    } catch (error) {
        console.error('Error al cargar el modelo:', error);
        showError('Error al cargar el modelo de IA. Por favor, recarga la página.');
        return false;
    }
}

async function loadPropuestasDatabase() {
    try {
        showLoadingStatus('Cargando propuestas...', 0);
        
        const response = await fetch('propuestas.json');
        if (!response.ok) {
            throw new Error('No se pudo cargar las propuestas');
        }
        propuestasDatabase = await response.json();
        console.log(`📚 Propuestas cargadas: ${propuestasDatabase.length}`);
        
        buildCategoryIndex();
        
        showLoadingStatus('Inicializando inteligencia artificial...', 5);
        
        const modelLoaded = await loadModel();
        if (!modelLoaded) return;
        
        showLoadingStatus('Verificando cache...', 35);
        const cachedEmbeddings = await loadEmbeddingsFromCache(propuestasDatabase);
        
        if (cachedEmbeddings) {
            embeddings = cachedEmbeddings;
            showLoadingStatus('✅ Cargado desde cache (instantáneo)', 100);
        } else {
            showLoadingStatus('Procesando propuestas (primera vez)...', 40);
            
            embeddings = [];
            const batchSize = 10;
            const totalBatches = Math.ceil(propuestasDatabase.length / batchSize);
            
            for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
                const start = batchIndex * batchSize;
                const end = Math.min(start + batchSize, propuestasDatabase.length);
                const batch = propuestasDatabase.slice(start, end);
                
                const batchEmbeddings = await Promise.all(
                    batch.map(async (item) => {
                        return await getEmbedding(item.propuesta);
                    })
                );
                
                embeddings.push(...batchEmbeddings);
                
                const progress = 40 + Math.floor(((end) / propuestasDatabase.length) * 55);
                showLoadingStatus(
                    `Procesando propuestas... (${end}/${propuestasDatabase.length})`,
                    progress
                );
            }
            
            showLoadingStatus('Guardando en cache...', 95);
            await saveEmbeddingsToCache(embeddings, propuestasDatabase);
        }
        
        showLoadingStatus('¡Todo listo! Puedes empezar a preguntar', 100);
        
        setTimeout(() => {
            const welcomeMsg = chatContainer.querySelector('.welcome-message');
            if (welcomeMsg) {
                const statusDiv = welcomeMsg.querySelector('.loading-status');
                if (statusDiv) {
                    statusDiv.style.opacity = '0';
                    statusDiv.style.transition = 'opacity 0.5s';
                    setTimeout(() => statusDiv.remove(), 500);
                }
            }
        }, 2000);
        
        console.log('✅ Sistema listo con', embeddings.length, 'embeddings');
        
    } catch (error) {
        console.error('Error al cargar propuestas:', error);
        showError('No se pudo cargar las propuestas.');
    }
}

// ====== BÚSQUEDA OPTIMIZADA ======

async function findBestMatch(query) {
    if (!isModelLoaded || embeddings.length === 0) {
        return {
            propuesta: 'El sistema aún está cargando. Por favor espera unos segundos.',
            similitud: 0,
            categoria: '',
            alternativas: []
        };
    }

    try {
        const queryEmbedding = await getEmbedding(query);
        
        const relevantIndices = getRelevantSubset(query);
        
        let matches = relevantIndices.map(index => {
            const similarity = cosineSimilarity(queryEmbedding, embeddings[index]);
            return {
                item: propuestasDatabase[index],
                similarity: similarity * 100
            };
        });

        matches.sort((a, b) => b.similarity - a.similarity);

        const bestMatch = matches[0];
        
        const alternativas = matches
            .slice(1, 4)
            .filter(match => match.similarity > 40)
            .map(match => ({
                propuesta: match.item.propuesta.substring(0, 150) + '...',
                similitud: match.similarity,
                categoria: match.item.categoria
            }));

        if (bestMatch.similarity < 30) {
            return {
                propuesta: 'No encontré información sobre eso en el plan de gobierno. Intenta preguntar sobre: asistencia social, pensiones, educación, salud, trabajo, cuido infantil, poblaciones vulnerables.',
                similitud: bestMatch.similarity,
                categoria: '',
                alternativas: alternativas
            };
        }

        return {
            propuesta: bestMatch.item.propuesta,
            similitud: bestMatch.similarity,
            categoria: bestMatch.item.categoria,
            alternativas: alternativas
        };
        
    } catch (error) {
        console.error('Error al buscar:', error);
        return {
            propuesta: 'Ocurrió un error al procesar tu pregunta. Por favor, intenta de nuevo.',
            similitud: 0,
            categoria: '',
            alternativas: []
        };
    }
}

// ====== UI ======

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    chatContainer.appendChild(errorDiv);
}

function addMessage(text, isUser, confidence = null, categoria = null) {
    const welcomeMsg = chatContainer.querySelector('.welcome-message');
    if (welcomeMsg && isUser) {
        welcomeMsg.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = isUser ? '👤' : '🤖';

    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = text;

    if (!isUser && categoria) {
        const categoryBadge = document.createElement('div');
        categoryBadge.className = 'category-badge';
        categoryBadge.textContent = `📂 ${categoria}`;
        categoryBadge.style.cssText = 'display: inline-block; background: #f0f0f0; padding: 4px 8px; border-radius: 4px; font-size: 11px; margin-bottom: 8px; color: #666;';
        content.insertBefore(categoryBadge, content.firstChild);
    }

    if (!isUser && confidence !== null) {
        const confidenceText = document.createElement('div');
        confidenceText.className = 'confidence';
        confidenceText.textContent = `Confianza: ${confidence.toFixed(1)}%`;
        content.appendChild(confidenceText);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);

    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot';
    loadingDiv.id = 'loading-indicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';

    const loadingContent = document.createElement('div');
    loadingContent.className = 'loading';
    loadingContent.innerHTML = '<span></span><span></span><span></span>';

    loadingDiv.appendChild(avatar);
    loadingDiv.appendChild(loadingContent);
    chatContainer.appendChild(loadingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeLoading() {
    const loadingIndicator = document.getElementById('loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.remove();
    }
}

async function handleUserQuestion() {
    const question = userInput.value.trim();

    if (question === '') return;

    if (!isModelLoaded) {
        addMessage('Por favor espera a que el sistema termine de cargar.', false);
        return;
    }

    addMessage(question, true);
    userInput.value = '';
    showLoading();

    const match = await findBestMatch(question);
    
    removeLoading();

    addMessage(match.propuesta, false, match.similitud, match.categoria);

    if (match.alternativas && match.alternativas.length > 0 && match.similitud < 80) {
        setTimeout(() => {
            let alternativasText = '💡 También encontré estas propuestas relacionadas:\n\n';
            match.alternativas.forEach((alt, index) => {
                alternativasText += `${index + 1}. [${alt.categoria}] ${alt.propuesta}\n\n`;
            });
            addMessage(alternativasText, false);
        }, 600);
    }

    if (match.similitud < 40) {
        setTimeout(() => {
            addMessage('💬 Tip: Intenta ser más específico. Puedo ayudarte con: salud, educación, pensiones, trabajo, cuido infantil, asistencia social, etc.', false);
        }, 1200);
    }
}

// Event listeners
sendBtn.addEventListener('click', handleUserQuestion);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleUserQuestion();
    }
});

loadPropuestasDatabase();