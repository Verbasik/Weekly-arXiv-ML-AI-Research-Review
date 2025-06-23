/**
 * mobile-enhancements.js
 * Улучшения мобильной версии для "Weekly arXiv ML/AI Research Review"
 */

(function() {
    'use strict';
    
    // ==================== КОНФИГУРАЦИЯ ====================
    const CONFIG = {
        // Пороги размеров экрана
        breakpoints: {
            mobile: 768,
            tablet: 992,
            smallPhone: 375
        },
        // Селекторы для основных элементов
        selectors: {
            nav: {
                container: '.nav-container',
                links: '.nav-links',
                dropdown: '.dropdown',
                dropdownContent: '.dropdown-content'
            },
            sidebar: {
                container: '.sidebar',
                headings: '.sidebar h3',
                lists: '.sidebar ul',
                divs: '.sidebar div'
            },
            modal: {
                container: '#markdown-modal',
                content: '.modal-content',
                closeButton: '.close-modal',
                markdownBody: '#markdown-content',
                loader: '.loader'
            }
        },
        // Классы для динамического добавления/удаления
        classes: {
            mobileActive: 'mobile-active',
            tapActive: 'tap-active',
            open: 'open',
            visible: 'visible',
            modalDragging: 'modal-dragging'
        }
    };
    
    // ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
    
    /**
     * Проверка размера экрана для мобильных устройств
     * @return {boolean} Возвращает true если ширина экрана меньше или равна порогу для мобильных
     */
    function isMobile() {
        return window.innerWidth <= CONFIG.breakpoints.mobile;
    }
    
    /**
     * Безопасное получение элемента DOM
     * @param {string} selector - CSS-селектор
     * @param {Element} [parent=document] - Родительский элемент для поиска
     * @return {Element|null} Найденный элемент или null
     */
    function getElement(selector, parent = document) {
        return parent.querySelector(selector);
    }
    
    /**
     * Безопасное получение коллекции элементов DOM
     * @param {string} selector - CSS-селектор
     * @param {Element} [parent=document] - Родительский элемент для поиска
     * @return {NodeList} Коллекция найденных элементов
     */
    function getElements(selector, parent = document) {
        return parent.querySelectorAll(selector);
    }
    
    /**
     * Создание элемента DOM с заданными свойствами
     * @param {string} tag - HTML-тег
     * @param {Object} [attributes={}] - Атрибуты элемента
     * @param {string} [innerHTML=''] - Внутреннее HTML-содержимое
     * @return {Element} Созданный элемент
     */
    function createElement(tag, attributes = {}, innerHTML = '') {
        const element = document.createElement(tag);
        
        // Установка атрибутов
        Object.entries(attributes).forEach(([key, value]) => {
            element.setAttribute(key, value);
        });
        
        // Установка внутреннего содержимого
        if (innerHTML) {
            element.innerHTML = innerHTML;
        }
        
        return element;
    }
    
    /**
     * Безопасное добавление обработчика события с защитой от дублирования
     * @param {Element} element - Элемент DOM
     * @param {string} eventType - Тип события
     * @param {Function} handler - Обработчик события
     * @param {Object} [options={}] - Дополнительные опции
     */
    function addSafeEventListener(element, eventType, handler, options = {}) {
        if (!element) return;
        
        // Клонируем элемент для удаления старых обработчиков
        const newElement = element.cloneNode(true);
        element.parentNode.replaceChild(newElement, element);
        
        // Добавляем новый обработчик
        newElement.addEventListener(eventType, handler, options);
        
        return newElement;
    }
    
    // ==================== ОСНОВНЫЕ ФУНКЦИИ ====================
    
    /**
     * Инициализация гамбургер-меню для мобильных устройств
     */
    function initMobileNav() {
        const navContainer = getElement(CONFIG.selectors.nav.container);
        const navLinks = getElement(CONFIG.selectors.nav.links);
        
        if (!navContainer || !navLinks) return;
        
        // Создаем гамбургер-меню только если его еще нет
        if (!getElement('.menu-toggle')) {
            const menuToggle = createElement('div', { class: 'menu-toggle' }, '☰');
            
            // Определяем первый элемент для вставки перед ним
            if (navContainer.firstChild) {
                navContainer.insertBefore(menuToggle, navContainer.firstChild);
            } else {
                navContainer.appendChild(menuToggle);
            }
            
            // Обработчик для гамбургер-меню
            menuToggle.addEventListener('click', function() {
                navLinks.classList.toggle(CONFIG.classes.mobileActive);
                // Меняем иконку при активации
                this.innerHTML = navLinks.classList.contains(CONFIG.classes.mobileActive) ? '✕' : '☰';
                
                // Если меню открыто, прокручиваем страницу вверх для лучшего UX
                if (navLinks.classList.contains(CONFIG.classes.mobileActive)) {
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                }
            });
        }
        
        // Обработка выпадающих списков в мобильной навигации
        initMobileDropdowns();
    }
    
    /**
     * Инициализация выпадающих списков для мобильных устройств
     */
    function initMobileDropdowns() {
        const dropdowns = getElements(CONFIG.selectors.nav.dropdown);
        
        dropdowns.forEach(dropdown => {
            const dropdownLink = getElement('a', dropdown);
            if (!dropdownLink) return;
            
            // Удаляем старые обработчики и добавляем новые
            const newDropdownLink = addSafeEventListener(dropdownLink, 'click', function(e) {
                if (isMobile()) {
                    e.preventDefault(); // Предотвращаем переход по ссылке
                    dropdown.classList.toggle(CONFIG.classes.tapActive);
                    
                    // Закрываем другие открытые выпадающие списки
                    dropdowns.forEach(otherDropdown => {
                        if (otherDropdown !== dropdown && otherDropdown.classList.contains(CONFIG.classes.tapActive)) {
                            otherDropdown.classList.remove(CONFIG.classes.tapActive);
                        }
                    });
                }
            });
        });
    }
    
    /**
     * Инициализация аккордеона для боковой панели на мобильных устройствах
     */
    function initSidebarAccordion() {
        if (!isMobile()) return;
        
        const sidebarHeadings = getElements(CONFIG.selectors.sidebar.headings);
        
        sidebarHeadings.forEach(heading => {
            // Проверяем, был ли уже установлен обработчик
            if (heading.getAttribute('data-accordion-initialized') === 'true') {
                return;
            }
            
            // Устанавливаем обработчик и помечаем как инициализированный
            const newHeading = addSafeEventListener(heading, 'click', function() {
                // Переключаем класс open для текущего заголовка
                this.classList.toggle(CONFIG.classes.open);
                
                // Если заголовок закрывается, закрываем и все остальные
                if (!this.classList.contains(CONFIG.classes.open)) {
                    sidebarHeadings.forEach(h => {
                        if (h !== this) {
                            h.classList.remove(CONFIG.classes.open);
                        }
                    });
                }
            });
            
            newHeading.setAttribute('data-accordion-initialized', 'true');
        });
    }
    
    /**
     * Улучшения для модальных окон на мобильных устройствах
     */
    function enhanceModals() {
        const modal = getElement(CONFIG.selectors.modal.container);
        if (!modal) return;
        
        // Получаем оригинальную функцию открытия модального окна из глобальной области
        const originalOpenReviewModal = window.openReviewModal;
        
        // Проверяем, что функция существует и еще не была улучшена
        if (originalOpenReviewModal && !window._enhancedModalBound) {
            window._enhancedModalBound = true;
            
            // Переопределяем функцию
            window.openReviewModal = function(year, week, title) {
                // Вызываем оригинальную функцию
                originalOpenReviewModal(year, week, title);
                
                // Улучшения для мобильных устройств
                const modalContent = getElement(CONFIG.selectors.modal.content, modal);
                
                // Блокируем скролл страницы при открытии модального окна
                document.body.style.overflow = 'hidden';
                
                // Добавляем индикатор для свайпа, если его еще нет
                if (!getElement('.modal-swipe-indicator', modalContent)) {
                    const swipeIndicator = createElement('div', { class: 'modal-swipe-indicator' });
                    modalContent.insertBefore(swipeIndicator, modalContent.firstChild);
                }
                
                // Устанавливаем мобильные жесты для закрытия модального окна
                if (isMobile()) {
                    setupModalSwipeGesture(modal, modalContent);
                }
                
                // Добавляем обработчик для кнопки закрытия
                const closeButton = getElement(CONFIG.selectors.modal.closeButton, modalContent);
                if (closeButton) {
                    closeButton.addEventListener('click', function() {
                        modal.style.display = 'none';
                        document.body.style.overflow = '';
                    });
                }
                
                // Улучшение доступности изображений
                enhanceMarkdownImages(modal);
            };
            
            // Добавляем обработчик для закрытия модального окна по клику вне содержимого
            modal.addEventListener('click', function(e) {
                if (e.target === modal) {
                    modal.style.display = 'none';
                    document.body.style.overflow = '';
                }
            });
        }
    }
    
    /**
     * Настройка жеста свайпа для закрытия модального окна
     * @param {Element} modal - Модальное окно
     * @param {Element} modalContent - Содержимое модального окна
     */
    function setupModalSwipeGesture(modal, modalContent) {
        if (!modal || !modalContent) return;
        
        let touchStartY = 0;
        let touchMoveY = 0;
        let isSwipeable = false;
        
        // Обработчик начала касания
        modalContent.addEventListener('touchstart', function(e) {
            const touch = e.touches[0];
            touchStartY = touch.clientY;
            
            // Определяем, можно ли свайпать (только если мы в верхней части контента)
            const contentTop = this.scrollTop;
            isSwipeable = contentTop <= 10;
            
            if (isSwipeable) {
                this.style.transition = 'transform 0.1s';
                modal.classList.add(CONFIG.classes.modalDragging);
            }
        }, { passive: true });
        
        // Обработчик движения пальца
        modalContent.addEventListener('touchmove', function(e) {
            if (!isSwipeable) return;
            
            const touch = e.touches[0];
            touchMoveY = touch.clientY - touchStartY;
            
            // Разрешаем жест только вниз
            if (touchMoveY > 0) {
                // Применяем затухание для естественного ощущения
                const dampenedMove = Math.pow(touchMoveY, 0.7);
                this.style.transform = `translateY(${dampenedMove}px)`;
                e.preventDefault();
            }
        }, { passive: false });
        
        // Обработчик окончания касания
        modalContent.addEventListener('touchend', function() {
            if (!isSwipeable) return;
            
            this.style.transition = 'transform 0.3s';
            modal.classList.remove(CONFIG.classes.modalDragging);
            
            // Если жест был достаточно длинным, закрываем модальное окно
            if (touchMoveY > 100) {
                modal.style.display = 'none';
                document.body.style.overflow = '';
            }
            
            // Возвращаем в исходное положение
            this.style.transform = '';
            touchStartY = 0;
            touchMoveY = 0;
            isSwipeable = false;
        }, { passive: true });
    }
    
    /**
     * Улучшение изображений в markdown для отзывчивости
     * @param {Element} modal - Модальное окно
     */
    function enhanceMarkdownImages(modal) {
        const markdownContent = getElement(CONFIG.selectors.modal.markdownBody, modal);
        if (!markdownContent) return;
        
        // Получаем все изображения внутри содержимого markdown
        const images = getElements('img', markdownContent);
        
        images.forEach(img => {
            // Устанавливаем стили для отзывчивости
            img.style.maxWidth = '100%';
            img.style.height = 'auto';
            
            // Добавляем атрибут loading="lazy" для ленивой загрузки
            img.setAttribute('loading', 'lazy');
            
            // Добавляем обработчик ошибки загрузки
            img.addEventListener('error', function() {
                this.style.display = 'none';
                
                // Создаем замещающий элемент для сообщения об ошибке
                const errorMsg = createElement('span', 
                    { class: 'img-error', style: 'color: #e74c3c; font-style: italic; display: block; margin: 1rem 0;' }, 
                    '[Ошибка загрузки изображения]'
                );
                
                this.parentNode.insertBefore(errorMsg, this);
            });
        });
    }
    
    /**
     * Ленивая загрузка изображений с использованием Intersection Observer
     */
    function lazyLoadImages() {
        if ('IntersectionObserver' in window) {
            // Создаем наблюдатель
            const imageObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        const dataSrc = img.getAttribute('data-src');
                        
                        if (dataSrc) {
                            // Устанавливаем src и удаляем data-src
                            img.setAttribute('src', dataSrc);
                            img.removeAttribute('data-src');
                            
                            // Прекращаем наблюдение за этим изображением
                            observer.unobserve(img);
                        }
                    }
                });
            });
            
            // Находим все изображения с атрибутом data-src
            const lazyImages = getElements('img[data-src]');
            lazyImages.forEach(img => {
                imageObserver.observe(img);
            });
        } else {
            // Fallback для браузеров без поддержки IntersectionObserver
            const lazyImages = getElements('img[data-src]');
            lazyImages.forEach(img => {
                img.setAttribute('src', img.getAttribute('data-src'));
                img.removeAttribute('data-src');
            });
        }
    }
    
    /**
     * Обработка изменения размера окна
     */
    function handleWindowResize() {
        const navLinks = getElement(CONFIG.selectors.nav.links);
        const menuToggle = getElement('.menu-toggle');
        
        // Сбрасываем состояние меню при переходе на десктоп
        if (!isMobile()) {
            if (navLinks) {
                navLinks.classList.remove(CONFIG.classes.mobileActive);
            }
            
            if (menuToggle) {
                menuToggle.innerHTML = '☰';
            }
            
            // Сбрасываем открытые выпадающие списки
            const openDropdowns = getElements(`.${CONFIG.classes.tapActive}`);
            openDropdowns.forEach(dropdown => {
                dropdown.classList.remove(CONFIG.classes.tapActive);
            });
        } else {
            // На мобильных устройствах инициализируем аккордеон для боковой панели
            initSidebarAccordion();
        }
        
        // Вызов инициализации мобильной навигации при изменении размера
        initMobileNav();
    }
    
    /**
     * Улучшение загрузки markdown-контента с рендерингом MathJax
     */
    function enhanceMarkdownRendering() {
        // Проверяем, существует ли оригинальная функция загрузки markdown
        const originalLoadMarkdownFromGitHub = window.loadMarkdownFromGitHub;
        
        if (originalLoadMarkdownFromGitHub && !window._enhancedMarkdownLoadingBound) {
            window._enhancedMarkdownLoadingBound = true;
            
            // Переопределяем функцию загрузки markdown
            window.loadMarkdownFromGitHub = async function(year, week) {
                // Вызываем оригинальную функцию и получаем результат
                const success = await originalLoadMarkdownFromGitHub(year, week);
                
                if (success) {
                    // Добавляем дополнительные улучшения для мобильных устройств
                    const markdownContent = getElement('#markdown-content');
                    
                    if (!markdownContent) return success;
                    
                    // Добавляем стили для таблиц на мобильных
                    const tables = getElements('table', markdownContent);
                    tables.forEach(table => {
                        table.style.display = 'block';
                        table.style.width = '100%';
                        table.style.overflowX = 'auto';
                    });
                    
                    // Улучшаем отображение кода на мобильных
                    const codeBlocks = getElements('pre code', markdownContent);
                    codeBlocks.forEach(code => {
                        // Добавляем горизонтальную прокрутку
                        const preElement = code.parentElement;
                        if (preElement) {
                            preElement.style.overflowX = 'auto';
                            preElement.style.webkitOverflowScrolling = 'touch';
                        }
                    });
                    
                    // Добавляем обработку LaTeX формул если есть MathJax
                    if (typeof MathJax !== 'undefined') {
                        if (MathJax.typesetPromise) {
                            await MathJax.typesetPromise([markdownContent]);
                        } else if (MathJax.Hub) {
                            MathJax.Hub.Queue(["Typeset", MathJax.Hub, markdownContent]);
                        }
                    }
                }
                
                return success;
            };
        }
    }
    
    /**
     * Начальная инициализация всех улучшений
     */
    function initAllEnhancements() {
        // Мобильная навигация
        initMobileNav();
        
        // Аккордеон для боковой панели
        initSidebarAccordion();
        
        // Улучшения для модальных окон
        enhanceModals();
        
        // Ленивая загрузка изображений
        lazyLoadImages();
        
        // Улучшение рендеринга markdown
        enhanceMarkdownRendering();
        
        // Добавление обработчика изменения размера окна
        window.addEventListener('resize', handleWindowResize);
        
        // Обработка изменений хеша URL для модальных окон
        if (window.checkUrlHash) {
            // Если есть оригинальная функция, используем её
            window.addEventListener('hashchange', window.checkUrlHash);
        }
        
        // Обработка 'popstate' событий для корректного закрытия модальных окон
        window.addEventListener('popstate', function() {
            const modal = getElement(CONFIG.selectors.modal.container);
            if (modal && modal.style.display === 'flex') {
                // Если URL изменился и модальное окно открыто, закрываем его
                if (!window.location.hash.includes('/')) {
                    modal.style.display = 'none';
                    document.body.style.overflow = '';
                }
            }
        });
        
        // Возвращаемся к нормальному скроллу при закрытии модального окна
        window.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                const modal = getElement(CONFIG.selectors.modal.container);
                if (modal && modal.style.display === 'flex') {
                    modal.style.display = 'none';
                    document.body.style.overflow = '';
                }
            }
        });
    }
    
    // ==================== ИНИЦИАЛИЗАЦИЯ ====================
    
    // Инициализируем все улучшения после загрузки DOM
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAllEnhancements);
    } else {
        // DOM уже загружен, запускаем инициализацию сейчас
        initAllEnhancements();
    }
    
    // Запускаем снова после полной загрузки страницы (для надежности)
    window.addEventListener('load', function() {
        // Проверяем, что DOM полностью загружен
        setTimeout(initAllEnhancements, 500);
    });
    
})();