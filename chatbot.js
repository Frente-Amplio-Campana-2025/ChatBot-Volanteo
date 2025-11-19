// Importar Transformers.js desde CDN
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Configuración: no usar cache local para evitar problemas
env.allowLocalModels = false;

// Variables globales
let qaDatabase = [];
let embeddings = [];
let extractor = null;
let isModelLoaded = false;

const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

// Configuración de cache
const CACHE_KEY = 'chatbot_embeddings_cache';
const CACHE_VERSION = 'v1.0'; // Cambia esto si actualizas las preguntas

// Función para mostrar el estado de carga
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

// Función para calcular similitud coseno entre dos vectores
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

// Función para generar embedding de un texto
async function getEmbedding(text) {
    if (!extractor) {
        throw new Error('El modelo no está cargado');
    }
    
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

// Función para guardar embeddings en IndexedDB
async function saveEmbeddingsToCache(embeddings, qaDatabase) {
    try {
        const cacheData = {
            version: CACHE_VERSION,
            timestamp: Date.now(),
            embeddings: embeddings,
            questions: qaDatabase.map(item => item.pregunta), // Solo preguntas para validar
            count: qaDatabase.length
        };
        
        localStorage.setItem(CACHE_KEY, JSON.stringify(cacheData));
        console.log('✅ Embeddings guardados en cache');
    } catch (error) {
        console.warn('⚠️ No se pudo guardar en cache:', error);
    }
}

// Función para cargar embeddings desde cache
async function loadEmbeddingsFromCache(qaDatabase) {
    try {
        const cached = localStorage.getItem(CACHE_KEY);
        if (!cached) return null;
        
        const cacheData = JSON.parse(cached);
        
        // Validar versión
        if (cacheData.version !== CACHE_VERSION) {
            console.log('🔄 Cache desactualizado, regenerando...');
            localStorage.removeItem(CACHE_KEY);
            return null;
        }
        
        // Validar cantidad de preguntas
        if (cacheData.count !== qaDatabase.length) {
            console.log('🔄 Cantidad de preguntas cambió, regenerando...');
            localStorage.removeItem(CACHE_KEY);
            return null;
        }
        
        // Validar que las preguntas sean las mismas
        const currentQuestions = qaDatabase.map(item => item.pregunta);
        const questionsMatch = currentQuestions.every((q, i) => q === cacheData.questions[i]);
        
        if (!questionsMatch) {
            console.log('🔄 Preguntas modificadas, regenerando...');
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

// Cargar el modelo de embeddings
async function loadModel() {
    try {
        showLoadingStatus('Descargando modelo de IA...', 10);
        
        // Usar modelo multilingüe optimizado para español
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

// Cargar el JSON y generar embeddings
async function loadQADatabase() {
    try {
        showLoadingStatus('Cargando base de datos...', 0);
        
        const response = await fetch('preguntas_respuestas.json');
        if (!response.ok) {
            throw new Error('No se pudo cargar la base de datos');
        }
        qaDatabase = await response.json();
        console.log(`Base de datos cargada: ${qaDatabase.length} preguntas`);
        
        showLoadingStatus('Inicializando inteligencia artificial...', 5);
        
        // Cargar el modelo
        const modelLoaded = await loadModel();
        if (!modelLoaded) return;
        
        // Intentar cargar desde cache
        showLoadingStatus('Verificando cache...', 35);
        const cachedEmbeddings = await loadEmbeddingsFromCache(qaDatabase);
        
        if (cachedEmbeddings) {
            // Usar embeddings del cache
            embeddings = cachedEmbeddings;
            showLoadingStatus('✅ Cargado desde cache (instantáneo)', 100);
        } else {
            // Generar embeddings nuevos
            showLoadingStatus('Procesando preguntas (primera vez)...', 40);
            
            embeddings = [];
            
            // Procesar en lotes para mejor rendimiento
            const batchSize = 10;
            const totalBatches = Math.ceil(qaDatabase.length / batchSize);
            
            for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
                const start = batchIndex * batchSize;
                const end = Math.min(start + batchSize, qaDatabase.length);
                const batch = qaDatabase.slice(start, end);
                
                // Procesar lote
                const batchEmbeddings = await Promise.all(
                    batch.map(async (item) => {
                        const combinedText = `${item.pregunta} ${item.respuesta}`;
                        return await getEmbedding(combinedText);
                    })
                );
                
                embeddings.push(...batchEmbeddings);
                
                // Actualizar progreso
                const progress = 40 + Math.floor(((end) / qaDatabase.length) * 55);
                showLoadingStatus(
                    `Procesando preguntas... (${end}/${qaDatabase.length})`,
                    progress
                );
            }
            
            // Guardar en cache
            showLoadingStatus('Guardando en cache...', 95);
            await saveEmbeddingsToCache(embeddings, qaDatabase);
        }
        
        showLoadingStatus('¡Todo listo! Puedes empezar a preguntar', 100);
        
        // Remover mensaje de carga después de 2 segundos
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
        showError('No se pudo cargar la base de datos. Verifica que el archivo preguntas_respuestas.json existe.');
    }
}

// Función para mostrar errores
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    chatContainer.appendChild(errorDiv);
}

// Función mejorada para encontrar las mejores coincidencias usando embeddings
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
        // Generar embedding de la pregunta del usuario
        const queryEmbedding = await getEmbedding(query);
        
        // Calcular similitud con todas las preguntas
        let allMatches = qaDatabase.map((item, index) => {
            const similarity = cosineSimilarity(queryEmbedding, embeddings[index]);
            return {
                item: item,
                similarity: similarity * 100 // Convertir a porcentaje
            };
        });

        // Ordenar por similitud descendente
        allMatches.sort((a, b) => b.similarity - a.similarity);

        // Obtener el mejor resultado
        const bestMatch = allMatches[0];
        
        // Obtener alternativas (top 3-5 con similitud > 40)
        const alternativas = allMatches
            .slice(1, 5)
            .filter(match => match.similarity > 40)
            .map(match => ({
                pregunta: match.item.pregunta,
                similitud: match.similarity
            }));

        // Si no hay coincidencia razonable (umbral más bajo con embeddings)
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

// Función para agregar mensaje al chat
function addMessage(text, isUser, confidence = null) {
    // Eliminar mensaje de bienvenida si existe
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

    // Agregar indicador de confianza para respuestas del bot
    if (!isUser && confidence !== null) {
        const confidenceText = document.createElement('div');
        confidenceText.className = 'confidence';
        confidenceText.textContent = `Confianza: ${confidence.toFixed(1)}%`;
        content.appendChild(confidenceText);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatContainer.appendChild(messageDiv);

    // Scroll automático al último mensaje
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Función para mostrar indicador de carga
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

// Función para eliminar indicador de carga
function removeLoading() {
    const loadingIndicator = document.getElementById('loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.remove();
    }
}

// Función para procesar la pregunta del usuario
async function handleUserQuestion() {
    const question = userInput.value.trim();

    if (question === '') {
        return;
    }

    if (!isModelLoaded) {
        addMessage('Por favor espera a que el sistema termine de cargar.', false);
        return;
    }

    // Agregar mensaje del usuario
    addMessage(question, true);

    // Limpiar input
    userInput.value = '';

    // Mostrar indicador de carga
    showLoading();

    // Buscar la mejor coincidencia
    const match = await findBestMatch(question);
    
    removeLoading();

    // Agregar respuesta del bot
    addMessage(match.respuesta, false, match.similitud);

    // Mostrar pregunta relacionada si la confianza es buena
    if (match.similitud >= 50 && match.pregunta) {
        setTimeout(() => {
            addMessage(`📌 Pregunta relacionada: "${match.pregunta}"`, false);
        }, 400);
    }

    // Si hay alternativas relevantes, mostrarlas
    if (match.alternativas && match.alternativas.length > 0 && match.similitud < 75) {
        setTimeout(() => {
            let alternativasText = '💡 También podrías preguntar sobre:\n\n';
            match.alternativas.slice(0, 3).forEach((alt, index) => {
                alternativasText += `${index + 1}. ${alt.pregunta}\n`;
            });
            addMessage(alternativasText, false);
        }, 800);
    }

    // Si la similitud es baja, dar sugerencias
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

// Cargar la base de datos al iniciar
loadQADatabase();