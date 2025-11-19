// Importar Transformers.js desde CDN
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

env.allowLocalModels = false;

// Variables globales
let qaDatabase = [];
let embeddings = [];
let categoryIndex = {}; // Índice por categorías
let extractor = null;
let isModelLoaded = false;

const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

const CACHE_KEY = 'chatbot_embeddings_cache';
const CACHE_VERSION = 'v2.0'; // Incrementar al cambiar estructura

// ====== SISTEMA DE CATEGORIZACIÓN ======

const CATEGORY_KEYWORDS = {
    'asistencia-social': ['imas', 'sinirube', 'pobreza', 'vulnerabilidad', 'asistencia', 'beneficiarios', 'fis'],
    'pensiones': ['pensión', 'pensiones', 'jubilación', 'adulto mayor', 'cuidadoras', '65 años', '60 años'],
    'educacion': ['avancemos', 'comedor', 'escolar', 'estudiante', 'beca', 'mep', 'secundaria', 'primaria'],
    'cuido': ['cecudi', 'cuido', 'cen-cinai', 'madres', 'infantil', 'red de cuido'],
    'trabajo': ['trabajo', 'laboral', 'fodesaf', 'inspección', 'patrono', 'salario', 'mtss'],
    'poblaciones-vulnerables': ['calle', 'abandono', 'iafa', 'psc', 'discapacidad', 'vulnerable'],
    'salud': ['salud', 'ccss', 'médico', 'hospital', 'ebais', 'ministerio de salud', 'comunidad'],
    'salud-mental': ['mental', 'adicción', 'drogas', 'ansiedad', 'depresión', 'suicida', 'eisaa'],
    'prevencion-salud': ['prevención', 'promoción', 'entorno', 'saludable', 'recreativo'],
    'reformas-legales': ['ley', 'reforma', 'proyecto', 'legal', 'reglamento'],
    'migracion': ['migrante', 'dimex', 'migración', 'extranjero']
};

// Detectar categorías relevantes de una consulta
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
    
    // Retornar categorías ordenadas por relevancia
    return Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .map(([cat]) => cat);
}

// Construir índice de categorías
function buildCategoryIndex() {
    categoryIndex = {};
    
    qaDatabase.forEach((item, index) => {
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

// Obtener subset de preguntas relevantes
function getRelevantSubset(query) {
    const categories = detectCategories(query);
    
    if (categories.length === 0) {
        // Si no hay categoría clara, buscar en todas
        return qaDatabase.map((_, idx) => idx);
    }
    
    // Buscar en categorías detectadas
    const indices = new Set();
    categories.slice(0, 3).forEach(cat => { // Top 3 categorías
        if (categoryIndex[cat]) {
            categoryIndex[cat].forEach(idx => indices.add(idx));
        }
    });
    
    // Si el subset es muy pequeño, agregar categoría general
    if (indices.size < 20 && categoryIndex['general']) {
        categoryIndex['general'].forEach(idx => indices.add(idx));
    }
    
    const result = Array.from(indices);
    console.log(`🎯 Búsqueda en ${result.length}/${qaDatabase.length} preguntas (${categories.join(', ')})`);
    
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

async function saveEmbeddingsToCache(embeddings, qaDatabase) {
    try {
        const cacheData = {
            version: CACHE_VERSION,
            timestamp: Date.now(),
            embeddings: embeddings,
            questions: qaDatabase.map(item => item.pregunta),
            count: qaDatabase.length
        };
        
        localStorage.setItem(CACHE_KEY, JSON.stringify(cacheData));
        console.log('✅ Embeddings guardados en cache');
    } catch (error) {
        console.warn('⚠️ No se pudo guardar en cache:', error);
    }
}

async function loadEmbeddingsFromCache(qaDatabase) {
    try {
        const cached = localStorage.getItem(CACHE_KEY);
        if (!cached) return null;
        
        const cacheData = JSON.parse(cached);
        
        if (cacheData.version !== CACHE_VERSION || 
            cacheData.count !== qaDatabase.length) {
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
        
        // Modelo optimizado para español
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

async function loadQADatabase() {
    try {
        showLoadingStatus('Cargando base de datos...', 0);
        
        const response = await fetch('preguntas_respuestas.json');
        if (!response.ok) {
            throw new Error('No se pudo cargar la base de datos');
        }
        qaDatabase = await response.json();
        console.log(`📚 Base de datos cargada: ${qaDatabase.length} preguntas`);
        
        // Construir índice de categorías
        buildCategoryIndex();
        
        showLoadingStatus('Inicializando inteligencia artificial...', 5);
        
        const modelLoaded = await loadModel();
        if (!modelLoaded) return;
        
        showLoadingStatus('Verificando cache...', 35);
        const cachedEmbeddings = await loadEmbeddingsFromCache(qaDatabase);
        
        if (cachedEmbeddings) {
            embeddings = cachedEmbeddings;
            showLoadingStatus('✅ Cargado desde cache (instantáneo)', 100);
        } else {
            showLoadingStatus('Procesando preguntas (primera vez)...', 40);
            
            embeddings = [];
            const batchSize = 10;
            const totalBatches = Math.ceil(qaDatabase.length / batchSize);
            
            for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
                const start = batchIndex * batchSize;
                const end = Math.min(start + batchSize, qaDatabase.length);
                const batch = qaDatabase.slice(start, end);
                
                const batchEmbeddings = await Promise.all(
                    batch.map(async (item) => {
                        const combinedText = `${item.pregunta} ${item.respuesta}`;
                        return await getEmbedding(combinedText);
                    })
                );
                
                embeddings.push(...batchEmbeddings);
                
                const progress = 40 + Math.floor(((end) / qaDatabase.length) * 55);
                showLoadingStatus(
                    `Procesando preguntas... (${end}/${qaDatabase.length})`,
                    progress
                );
            }
            
            showLoadingStatus('Guardando en cache...', 95);
            await saveEmbeddingsToCache(embeddings, qaDatabase);
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
        console.error('Error al cargar la base de datos:', error);
        showError('No se pudo cargar la base de datos.');
    }
}

// ====== BÚSQUEDA OPTIMIZADA ======

async function findBestMatch(query) {
    if (!isModelLoaded || embeddings.length === 0) {
        return {
            respuesta: 'El sistema aún está cargando. Por favor espera unos segundos.',
            similitud: 0,
            pregunta: '',
            alternativas: []
        };
    }

    try {
        const queryEmbedding = await getEmbedding(query);
        
        // OPTIMIZACIÓN: Buscar solo en subset relevante
        const relevantIndices = getRelevantSubset(query);
        
        // Calcular similitud solo con preguntas relevantes
        let matches = relevantIndices.map(index => {
            const similarity = cosineSimilarity(queryEmbedding, embeddings[index]);
            return {
                item: qaDatabase[index],
                similarity: similarity * 100
            };
        });

        matches.sort((a, b) => b.similarity - a.similarity);

        const bestMatch = matches[0];
        
        const alternativas = matches
            .slice(1, 5)
            .filter(match => match.similarity > 40)
            .map(match => ({
                pregunta: match.item.pregunta,
                similitud: match.similarity
            }));

        if (bestMatch.similarity < 30) {
            return {
                respuesta: 'No encontré una respuesta directa a tu pregunta. Intenta reformular o usar términos como: "salud", "CCSS", "mujeres", "violencia", "trabajo", "cannabis", "licencia menstrual", "deuda", "pensiones".',
                similitud: bestMatch.similarity,
                pregunta: '',
                alternativas: alternativas
            };
        }

        return {
            respuesta: bestMatch.item.respuesta,
            similitud: bestMatch.similarity,
            pregunta: bestMatch.item.pregunta,
            alternativas: alternativas
        };
        
    } catch (error) {
        console.error('Error al buscar coincidencia:', error);
        return {
            respuesta: 'Ocurrió un error al procesar tu pregunta. Por favor, intenta de nuevo.',
            similitud: 0,
            pregunta: '',
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

function addMessage(text, isUser, confidence = null) {
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

    addMessage(match.respuesta, false, match.similitud);

    if (match.similitud >= 50 && match.pregunta) {
        setTimeout(() => {
            addMessage(`📌 Pregunta relacionada: "${match.pregunta}"`, false);
        }, 400);
    }

    if (match.alternativas && match.alternativas.length > 0 && match.similitud < 75) {
        setTimeout(() => {
            let alternativasText = '💡 También podrías preguntar sobre:\n\n';
            match.alternativas.slice(0, 3).forEach((alt, index) => {
                alternativasText += `${index + 1}. ${alt.pregunta}\n`;
            });
            addMessage(alternativasText, false);
        }, 800);
    }

    if (match.similitud < 40) {
        setTimeout(() => {
            addMessage('💬 Tip: Intenta ser más específico. Puedo ayudarte con temas como: salud, CCSS, mujeres, violencia, trabajo, cannabis, pensiones, etc.', false);
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

loadQADatabase();