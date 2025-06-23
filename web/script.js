// script.js

// Конфигурация GitHub
const GITHUB_REPO = 'Verbasik/Weekly-arXiv-ML-AI-Research-Review';
const GITHUB_BRANCH = 'develop'; // Или 'main', если ветка называется так

// ==================== УЛУЧШЕННЫЙ ERROR HANDLING ==================== 

// Конфигурация для retry механизма
const RETRY_CONFIG = {
    maxRetries: 3,
    baseDelay: 1000, // 1 секунда
    maxDelay: 10000, // 10 секунд
    backoffFactor: 2
};

// Утилиты для error handling
const ErrorHandler = {
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
async function fetchWithRetry(url, options = {}, context = '') {
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
function createErrorUI(errorType, context = '', onRetry = null, onBack = null) {
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

// Получение основных элементов DOM
const contentElement = document.querySelector('.content');
const modal = document.getElementById('markdown-modal');
const markdownContent = document.getElementById('markdown-content');
const closeModalButton = modal ? modal.querySelector('.close-modal') : null;
const loader = modal ? modal.querySelector('.loader') : null;
const backToTopButton = document.getElementById('back-to-top');
const searchInput = document.querySelector('.search-bar input');
const searchButton = document.querySelector('.search-bar button');

// --- Загрузка и отображение данных ---

async function loadWeeksData() {
    const jsonUrl = `https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/web/index.json`;

    if (!contentElement) {
        console.error("Content element not found.");
        return;
    }

    // Показываем индикатор загрузки
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'loading-indicator';
    loadingIndicator.innerHTML = `
        <div class="loader"></div>
        <p>Загрузка статей...</p>
    `;
    contentElement.appendChild(loadingIndicator);

    try {
        const response = await fetchWithRetry(jsonUrl, {}, 'данные статей');
        const data = await response.json();

        // Удаляем индикатор загрузки
        loadingIndicator.remove();

        // Очистка старых секций
        contentElement.querySelectorAll('.year-section:not(#home)').forEach(section => section.remove());

        // Создание секций и карточек
        data.years.forEach(yearData => {
            const yearSection = createYearSection(yearData.year, contentElement);
            yearData.weeks.forEach(weekData => {
                createWeekCard(yearSection, yearData.year, weekData);
            });
        });

        updateYearFilters(data.years.map(y => y.year));
        checkUrlHash(); // Проверяем хэш после загрузки и рендеринга

    } catch (error) {
        console.error('Error loading weeks data:', error);
        
        // Удаляем индикатор загрузки
        loadingIndicator.remove();
        
        // Определяем тип ошибки
        const errorInfo = ErrorHandler.classifyError(error);
        
        // Создаем улучшенный error UI
        const errorUI = createErrorUI(
            errorInfo.type, 
            'данные статей',
            () => {
                // Retry callback
                errorUI.remove();
                loadWeeksData();
            },
            null // Нет кнопки "Назад" для главной загрузки
        );
        
        contentElement.appendChild(errorUI);
        
        // Автоматическая попытка перезагрузки при восстановлении соединения
        if (errorInfo.type === 'offline') {
            const handleOnline = () => {
                errorUI.remove();
                loadWeeksData();
                window.removeEventListener('online', handleOnline);
            };
            window.addEventListener('online', handleOnline);
        }
    }
}

function createYearSection(year, parentElement) {
    const yearSection = document.createElement('section');
    yearSection.id = year;
    yearSection.className = 'year-section';
    yearSection.innerHTML = `
        <h2 class="year-title section-heading">${year} Papers</h2>
        <div class="weeks-grid"></div>
    `;
    parentElement.appendChild(yearSection);
    return yearSection;
}

function createWeekCard(yearSection, year, weekData) {
    const weeksGrid = yearSection.querySelector('.weeks-grid');
    if (!weeksGrid) return;

    const card = document.createElement('div');
    card.className = 'week-card';
    card.setAttribute('data-week', weekData.week);
    card.setAttribute('data-year', year);

    const tagsHtml = weekData.tags?.map(tag => `<span class="mono"><i class="fas fa-tag"></i> ${tag}</span>`).join('') || '';
    const dateHtml = weekData.date ? `<span><i class="far fa-calendar"></i> ${weekData.date}</span>` : '';

    const footerItems = [];

    // Papers
    if (weekData.papers !== undefined) {
        const paperText = `${weekData.papers} Paper${weekData.papers !== 1 ? 's' : ''}`;
        footerItems.push(`<span><i class="far fa-file-alt"></i> ${paperText}</span>`);
    }

    // Notebooks
    if (weekData.notebooks !== undefined && weekData.notebook_path) {
        const notebooksText = `${weekData.notebooks} Notebook${weekData.notebooks !== 1 ? 's' : ''}`;
        const notebookUrl = `https://github.com/${GITHUB_REPO}/tree/${GITHUB_BRANCH}/${weekData.notebook_path}`;
        footerItems.push(`<a href="${notebookUrl}" target="_blank"><i class="far fa-file-code"></i> ${notebooksText}</a>`);
    } else if (weekData.notebooks !== undefined) {
        const notebooksText = `${weekData.notebooks} Notebook${weekData.notebooks !== 1 ? 's' : ''}`;
        footerItems.push(`<span><i class="far fa-file-code"></i> ${notebooksText}</span>`);
    }

    // Code files
    if (weekData.code !== undefined && weekData.code_path) {
        const codeText = `${weekData.code} Code${weekData.code !== 1 ? ' files' : ''}`;
        const codeUrl = `https://github.com/${GITHUB_REPO}/tree/${GITHUB_BRANCH}/${weekData.code_path}`;
        footerItems.push(`<a href="${codeUrl}" target="_blank"><i class="fas fa-code"></i> ${codeText}</a>`);
    } else if (weekData.code !== undefined) {
        const codeText = `${weekData.code} Code${weekData.code !== 1 ? ' files' : ''}`;
        footerItems.push(`<span><i class="fas fa-code"></i> ${codeText}</span>`);
    }

    card.innerHTML = `
        <div class="week-card-header">
            <h3 class="week-card-title">${weekData.title}</h3>
        </div>
        <div class="week-card-body">
            <div class="week-card-meta">${dateHtml} ${tagsHtml}</div>
            <p class="week-card-desc">${weekData.description || 'No description available.'}</p>
            <button class="gradient-button read-review">Read Review</button>
        </div>
        <div class="week-card-footer">
            ${footerItems.join('')}
        </div>
    `;

    weeksGrid.appendChild(card);

    // Обработчик для кнопки "Read Review"
    card.querySelector('.read-review')?.addEventListener('click', (e) => {
        e.preventDefault();
        openReviewModal(year, weekData.week, weekData.title);
    });
}

function updateYearFilters(years) {
    const yearFilterList = document.querySelector('.sidebar ul:first-of-type');
    if (!yearFilterList) return;

    yearFilterList.innerHTML = '';
    years.sort((a, b) => b - a).forEach((year, index) => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = `#${year}`;
        a.textContent = year;
        if (index === 0) a.className = 'active'; // Первый год активный по умолчанию
        li.appendChild(a);
        yearFilterList.appendChild(li);

        // Обработчик клика для фильтра
        a.addEventListener('click', function(e) {
            // Не предотвращаем стандартное поведение якоря (e.preventDefault()),
            // чтобы URL обновлялся и можно было использовать history.back/forward.
            // Просто обновляем активный класс.
            yearFilterList.querySelectorAll('a').forEach(link => link.classList.remove('active'));
            this.classList.add('active');
            // Плавная прокрутка к секции (опционально)
            // const targetElement = document.getElementById(this.getAttribute('href').substring(1));
            // if (targetElement) targetElement.scrollIntoView({ behavior: 'smooth' });
        });
    });
}

// --- Модальное окно и рендеринг Markdown/MathJax ---

async function loadMarkdownFromGitHub(year, week) {
    const reviewUrl = `https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/${year}/${week}/review.md`;

    if (!markdownContent || !loader) {
        console.error("Markdown content area or loader not found.");
        return false;
    }

    // Улучшенный индикатор загрузки
    loader.style.display = 'block';
    markdownContent.innerHTML = `
        <div class="loading-content">
            <div class="loader"></div>
            <p>Загрузка статьи "${year}/${week}"...</p>
            <p class="loading-tip">💡 Обычно это занимает несколько секунд</p>
        </div>
    `;

    try {
        // Используем улучшенную функцию fetch с retry
        const response = await fetchWithRetry(reviewUrl, {}, `статья "${year}/${week}"`);
        let markdown = await response.text();

        // Проверяем, что markdown не пустой
        if (!markdown.trim()) {
            throw new Error('Статья пуста или не содержит контента');
        }

        // 1. Изоляция формул MathJax
        const mathPlaceholders = {};
        let placeholderId = 0;
        const mathRegex = /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\$(?:[^$\\]|\\.)*?\$|\\\((?:[^)\\]|\\.)*?\\\))/g;
        markdown = markdown.replace(mathRegex, (match) => {
            const id = `mathjax-placeholder-${placeholderId++}`;
            mathPlaceholders[id] = match;
            return `<span id="${id}" style="display: none;"></span>`; // Плейсхолдер
        });

        // 2. Преобразование Markdown в HTML
        if (typeof marked === 'undefined') {
            throw new Error("Библиотека Marked.js не загружена. Попробуйте обновить страницу.");
        }
        
        let html;
        try {
            html = marked.parse(markdown);
        } catch (markdownError) {
            throw new Error(`Ошибка обработки Markdown: ${markdownError.message}`);
        }

        // 3. Вставка HTML
        markdownContent.innerHTML = html;

        // 4. Восстановление формул
        Object.keys(mathPlaceholders).forEach(id => {
            const placeholderElement = markdownContent.querySelector(`#${id}`);
            if (placeholderElement) {
                placeholderElement.replaceWith(document.createTextNode(mathPlaceholders[id]));
            }
        });

        // 5. Рендеринг MathJax
        try {
            if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
                MathJax.texReset?.();
                MathJax.typesetClear?.([markdownContent]);
                await MathJax.typesetPromise([markdownContent]);
            } else {
                console.warn("MathJax 3 not found or not configured.");
            }
        } catch (mathJaxError) {
            console.warn("MathJax rendering failed:", mathJaxError);
            // Не выбрасываем ошибку, так как статья может отображаться без формул
        }

        loader.style.display = 'none';
        return true;

    } catch (error) {
        console.error('Error loading or processing markdown:', error);
        
        // Определяем тип ошибки
        const errorInfo = ErrorHandler.classifyError(error);
        
        // Создаем улучшенный error UI для модального окна
        const errorUI = createErrorUI(
            errorInfo.type,
            `статья "${year}/${week}"`,
            () => {
                // Retry callback
                loadMarkdownFromGitHub(year, week);
            },
            () => {
                // Back callback - закрываем модальное окно
                closeModal();
            }
        );
        
        markdownContent.innerHTML = '';
        markdownContent.appendChild(errorUI);
        loader.style.display = 'none';
        
        // Автоматическая попытка перезагрузки при восстановлении соединения
        if (errorInfo.type === 'offline') {
            const handleOnline = () => {
                loadMarkdownFromGitHub(year, week);
                window.removeEventListener('online', handleOnline);
            };
            window.addEventListener('online', handleOnline);
        }
        
        return false;
    }
}

async function openReviewModal(year, week, title) {
    if (!modal || !markdownContent) return;

    // Установка заголовка
    const modalContentDiv = modal.querySelector('.modal-content');
    let titleElement = modalContentDiv?.querySelector('h2.modal-title');
    if (!titleElement) {
        titleElement = document.createElement('h2');
        titleElement.className = 'modal-title';
        titleElement.style.marginTop = '0';
        titleElement.style.marginBottom = '1rem';
        modalContentDiv?.insertBefore(titleElement, markdownContent);
    }
    titleElement.textContent = title;

    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    const success = await loadMarkdownFromGitHub(year, week);

    if (success) {
        // Обновляем хэш только при успехе
        window.location.hash = `#${year}/${week}`;
    }
    // Не сбрасываем хэш при ошибке, чтобы пользователь видел, что пытался открыть
}

function checkUrlHash() {
    const hash = window.location.hash;
    if (hash && hash.startsWith('#') && hash.includes('/')) {
        const parts = hash.substring(1).split('/');
        if (parts.length === 2 && parts[0] && parts[1]) {
            const year = parts[0];
            const week = parts[1];

            // Ищем карточку для заголовка (может не найтись, если DOM еще не готов)
            const card = document.querySelector(`.week-card[data-year="${year}"][data-week="${week}"]`);
            const title = card?.querySelector('.week-card-title')?.textContent || `Review ${year}/${week}`;

            // Открываем модальное окно, если оно еще не открыто для этого хэша
            const currentModalTitle = modal?.querySelector('.modal-content h2.modal-title');
            if (modal && (modal.style.display !== 'flex' || !currentModalTitle || currentModalTitle.textContent !== title)) {
                 openReviewModal(year, week, title);
            }
        } else {
             // Хэш не соответствует формату, закрываем окно
             if (modal && modal.style.display === 'flex') {
                 closeModal();
             }
        }
    } else {
        // Хэш пуст или не содержит '/', закрываем окно
        if (modal && modal.style.display === 'flex') {
            closeModal();
        }
    }
}

function closeModal() {
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = '';
        if (markdownContent) markdownContent.innerHTML = ''; // Очищаем контент
        // Сбрасываем хэш, чтобы при обновлении страницы окно не открылось снова
        history.pushState("", document.title, window.location.pathname + window.location.search);
    }
}

// --- Обработчики событий ---

// Закрытие модального окна
closeModalButton?.addEventListener('click', closeModal);
window.addEventListener('click', (event) => {
    if (event.target === modal) {
        closeModal();
    }
});
window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && modal && modal.style.display === 'flex') {
        closeModal();
    }
});


// Кнопка "Наверх"
if (backToTopButton) {
    window.addEventListener('scroll', () => {
        backToTopButton.classList.toggle('visible', window.pageYOffset > 300);
    });
    backToTopButton.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

// Поиск (заглушка)
function performSearch(query) {
    if (!query) return;
    alert(`Search functionality is not implemented. You searched for: ${query}`);
    // TODO: Реализовать логику поиска/фильтрации карточек
}

searchButton?.addEventListener('click', () => {
    if (searchInput) performSearch(searchInput.value.trim());
});

searchInput?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && searchInput) {
        performSearch(searchInput.value.trim());
    }
});

// ==================== NETWORK STATUS MONITORING ====================

// Создание индикатора статуса сети
function createNetworkStatusIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'network-status online';
    indicator.innerHTML = '🌐 Онлайн';
    document.body.appendChild(indicator);
    return indicator;
}

// Управление статусом сети
function initNetworkMonitoring() {
    const indicator = createNetworkStatusIndicator();
    
    // Обработчики событий сети
    window.addEventListener('online', () => {
        indicator.className = 'network-status online';
        indicator.innerHTML = '🌐 Подключение восстановлено';
        
        // Скрываем индикатор через 3 секунды
        setTimeout(() => {
            indicator.style.opacity = '0';
        }, 3000);
        
        console.log('Network connection restored');
    });
    
    window.addEventListener('offline', () => {
        indicator.className = 'network-status offline';
        indicator.innerHTML = '📡 Нет подключения';
        indicator.style.opacity = '1';
        
        console.log('Network connection lost');
    });
    
    // Проверяем начальное состояние
    if (!navigator.onLine) {
        indicator.className = 'network-status offline';
        indicator.innerHTML = '📡 Нет подключения';
        indicator.style.opacity = '1';
    }
}

// ==================== УЛУЧШЕННАЯ ИНИЦИАЛИЗАЦИЯ ====================

// Функция инициализации с error handling
async function initializeApp() {
    try {
        // Инициализируем мониторинг сети
        initNetworkMonitoring();
        
        // Загружаем данные
        await loadWeeksData();
        
        // Проверяем URL hash
        checkUrlHash();
        
        console.log('Application initialized successfully');
        
    } catch (error) {
        console.error('Failed to initialize application:', error);
        
        // Показываем критическую ошибку
        if (contentElement) {
            const criticalError = createErrorUI(
                'unknown',
                'приложение',
                () => {
                    window.location.reload();
                },
                null
            );
            
            contentElement.innerHTML = '';
            contentElement.appendChild(criticalError);
        }
    }
}

// Улучшенная функция обработки поиска
function performSearch(query) {
    if (!query) return;
    
    // Временная заглушка с улучшенным UX
    const searchModal = document.createElement('div');
    searchModal.className = 'modal';
    searchModal.style.display = 'flex';
    
    searchModal.innerHTML = `
        <div class="modal-content" style="max-width: 500px;">
            <span class="close-modal">×</span>
            <h2>🔍 Поиск</h2>
            <p>Функция поиска находится в разработке.</p>
            <p><strong>Ваш запрос:</strong> "${query}"</p>
            <div style="margin-top: 20px;">
                <p><strong>Пока что вы можете:</strong></p>
                <ul style="text-align: left; margin-top: 10px;">
                    <li>Просматривать статьи по годам в боковой панели</li>
                    <li>Использовать теги для фильтрации</li>
                    <li>Просматривать популярные статьи</li>
                </ul>
            </div>
            <button class="gradient-button" onclick="this.closest('.modal').remove()">Понятно</button>
        </div>
    `;
    
    document.body.appendChild(searchModal);
    
    // Обработчик закрытия
    searchModal.querySelector('.close-modal').addEventListener('click', () => {
        searchModal.remove();
    });
    
    // Закрытие по клику вне модального окна
    searchModal.addEventListener('click', (e) => {
        if (e.target === searchModal) {
            searchModal.remove();
        }
    });
}

// --- Инициализация ---

// Используем улучшенную инициализацию
window.addEventListener('DOMContentLoaded', initializeApp);
window.addEventListener('hashchange', checkUrlHash);

// Добавляем обработчик для unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    
    // Предотвращаем показ ошибки в консоли браузера
    event.preventDefault();
    
    // Можно добавить уведомление пользователю о проблеме
    const notification = document.createElement('div');
    notification.className = 'error-notification';
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(244, 67, 54, 0.9);
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 14px;
        z-index: 10000;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    `;
    notification.innerHTML = '⚠️ Произошла неожиданная ошибка';
    
    document.body.appendChild(notification);
    
    // Удаляем уведомление через 5 секунд
    setTimeout(() => {
        notification.remove();
    }, 5000);
});