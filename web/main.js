import { ResearchController } from './application/ResearchController.js';

/**
 * Main Application Entry Point
 * Точка входа в приложение с DDD архитектурой
 */

// Конфигурация приложения
const APP_CONFIG = {
    githubRepo: 'Verbasik/Weekly-arXiv-ML-AI-Research-Review',
    githubBranch: 'develop' // Или 'main', если ветка называется так
};

// Главная функция инициализации
async function initializeApplication() {
    try {
        // Создаем контроллер приложения
        const researchController = new ResearchController(APP_CONFIG);
        
        // Инициализируем приложение
        await researchController.initialize();
        
    } catch (error) {
        console.error('❌ Failed to initialize application:', error);
        
        // Показываем критическую ошибку пользователю
        showCriticalError(error);
    }
}

/**
 * Показывает критическую ошибку при инициализации
 */
function showCriticalError(error) {
    const contentElement = document.querySelector('.content');
    if (!contentElement) return;
    
    contentElement.innerHTML = `
        <div class="error-message enhanced" style="margin: 2rem auto; max-width: 600px;">
            <div class="error-header">
                <h4>⚠️ Критическая ошибка приложения</h4>
            </div>
            <div class="error-body">
                <p>Не удалось инициализировать приложение. Попробуйте обновить страницу.</p>
                <p style="font-size: 14px; color: #9CA3AF; margin-top: 1rem;">
                    Техническая информация: ${error.message}
                </p>
            </div>
            <div class="error-actions">
                <button class="gradient-button" onclick="window.location.reload()">
                    🔄 Обновить страницу
                </button>
            </div>
        </div>
    `;
}

// Инициализация при загрузке DOM
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApplication);
} else {
    // DOM уже загружен
    initializeApplication();
}

// Запускаем снова после полной загрузки страницы (для надежности)
window.addEventListener('load', function() {
    // Проверяем, что DOM полностью загружен
    setTimeout(() => {
        // Если приложение еще не инициализировано, пробуем еще раз
        if (!document.querySelector('.year-section')) {
            initializeApplication();
        }
    }, 500);
});

// Экспортируем для глобального доступа (если нужно)
window.ResearchApp = {
    initialize: initializeApplication,
    config: APP_CONFIG
}; 