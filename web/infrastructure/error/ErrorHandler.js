/**
 * Error Handler - Infrastructure Layer
 * Централизованная обработка ошибок
 */

// Конфигурация для retry механизма
const RETRY_CONFIG = {
    maxRetries: 3,
    baseDelay: 1000, // 1 секунда
    maxDelay: 10000, // 10 секунд
    backoffFactor: 2
};

// Утилиты для error handling
export const ErrorHandler = {
    // Проверка подключения к интернету
    isOnline: () => navigator.onLine,
    
    // Определение типа ошибки
    classifyError: (error, response = null) => {
        if (!navigator.onLine) {
            return { type: 'offline', severity: 'high', retryable: true };
        }
        
        if (response) {
            if (response.status === 404) {
                return { type: 'not_found', severity: 'medium', retryable: false };
            }
            if (response.status >= 500) {
                return { type: 'server_error', severity: 'high', retryable: true };
            }
            if (response.status === 403) {
                return { type: 'forbidden', severity: 'medium', retryable: false };
            }
            if (response.status >= 400) {
                return { type: 'client_error', severity: 'medium', retryable: false };
            }
        }
        
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            return { type: 'network', severity: 'high', retryable: true };
        }
        
        return { type: 'unknown', severity: 'medium', retryable: true };
    },
    
    // Генерация пользовательских сообщений
    getUserMessage: (errorType, context = '') => {
        const messages = {
            offline: {
                title: '🌐 Нет подключения к интернету',
                message: 'Проверьте подключение к сети и попробуйте снова.',
                action: 'Повторить попытку'
            },
            not_found: {
                title: '📄 Контент не найден',
                message: `${context} не найден. Возможно, он еще не опубликован или был перемещен.`,
                action: 'Вернуться к списку'
            },
            server_error: {
                title: '⚠️ Ошибка сервера',
                message: 'Временные проблемы с сервером. Мы работаем над их устранением.',
                action: 'Повторить попытку'
            },
            forbidden: {
                title: '🔒 Доступ ограничен',
                message: 'У вас нет прав для просмотра этого контента.',
                action: 'Вернуться к списку'
            },
            client_error: {
                title: '❌ Ошибка запроса',
                message: 'Произошла ошибка при загрузке данных.',
                action: 'Повторить попытку'
            },
            network: {
                title: '🌐 Проблемы с сетью',
                message: 'Не удается подключиться к серверу. Проверьте интернет-соединение.',
                action: 'Повторить попытку'
            },
            unknown: {
                title: '⚠️ Неизвестная ошибка',
                message: 'Произошла непредвиденная ошибка. Попробуйте обновить страницу.',
                action: 'Повторить попытку'
            }
        };
        
        return messages[errorType] || messages.unknown;
    },
    
    // Задержка с экспоненциальным backoff
    delay: (attempt) => {
        const delay = Math.min(
            RETRY_CONFIG.baseDelay * Math.pow(RETRY_CONFIG.backoffFactor, attempt),
            RETRY_CONFIG.maxDelay
        );
        return new Promise(resolve => setTimeout(resolve, delay));
    }
};

// Улучшенная функция fetch с retry
export async function fetchWithRetry(url, options = {}, context = '') {
    let lastError = null;
    let lastResponse = null;
    
    for (let attempt = 0; attempt <= RETRY_CONFIG.maxRetries; attempt++) {
        try {
            // Проверяем подключение перед запросом
            if (!ErrorHandler.isOnline()) {
                throw new Error('No internet connection');
            }
            
            // Создаем AbortController для timeout (совместимость с старыми браузерами)
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 секунд timeout
            
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            lastResponse = response;
            
            if (!response.ok) {
                const errorInfo = ErrorHandler.classifyError(null, response);
                
                // Если ошибка не требует retry, выбрасываем сразу
                if (!errorInfo.retryable) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                // Если это последняя попытка, выбрасываем ошибку
                if (attempt === RETRY_CONFIG.maxRetries) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                // Ждем перед следующей попыткой
                console.warn(`Attempt ${attempt + 1} failed, retrying in ${RETRY_CONFIG.baseDelay * Math.pow(RETRY_CONFIG.backoffFactor, attempt)}ms...`);
                await ErrorHandler.delay(attempt);
                continue;
            }
            
            return response;
            
        } catch (error) {
            lastError = error;
            
            // Если это AbortError (timeout), классифицируем как network error
            if (error.name === 'AbortError') {
                lastError = new Error('Request timeout');
            }
            
            const errorInfo = ErrorHandler.classifyError(lastError, lastResponse);
            
            // Если ошибка не требует retry или это последняя попытка
            if (!errorInfo.retryable || attempt === RETRY_CONFIG.maxRetries) {
                throw lastError;
            }
            
            // Ждем перед следующей попыткой
            console.warn(`Attempt ${attempt + 1} failed: ${lastError.message}, retrying in ${RETRY_CONFIG.baseDelay * Math.pow(RETRY_CONFIG.backoffFactor, attempt)}ms...`);
            await ErrorHandler.delay(attempt);
        }
    }
    
    throw lastError;
}

// Создание улучшенного error UI
export function createErrorUI(errorType, context = '', onRetry = null, onBack = null) {
    const userMessage = ErrorHandler.getUserMessage(errorType, context);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message enhanced';
    
    errorDiv.innerHTML = `
        <div class="error-header">
            <h4>${userMessage.title}</h4>
        </div>
        <div class="error-body">
            <p>${userMessage.message}</p>
            ${!ErrorHandler.isOnline() ? '<p class="offline-notice">📡 Ожидание подключения к интернету...</p>' : ''}
        </div>
        <div class="error-actions">
            ${onRetry ? `<button class="gradient-button retry-button">${userMessage.action}</button>` : ''}
            ${onBack ? '<button class="gradient-button secondary back-button">← Назад к списку</button>' : ''}
        </div>
    `;
    
    // Добавляем обработчики событий
    if (onRetry) {
        const retryButton = errorDiv.querySelector('.retry-button');
        retryButton?.addEventListener('click', onRetry);
    }
    
    if (onBack) {
        const backButton = errorDiv.querySelector('.back-button');
        backButton?.addEventListener('click', onBack);
    }
    
    return errorDiv;
} 