// script.js

// Конфигурация GitHub
const GITHUB_REPO = 'Verbasik/Weekly-arXiv-ML-AI-Research-Review';
const GITHUB_BRANCH = 'master'; // Или 'main', если ветка называется так

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

    try {
        const response = await fetch(jsonUrl);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();

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
        contentElement.innerHTML += `<div class="error-message"><p>Could not load data. Error: ${error.message}</p></div>`;
    }
}

function createYearSection(year, parentElement) {
    const yearSection = document.createElement('section');
    yearSection.id = year;
    yearSection.className = 'year-section';
    yearSection.innerHTML = `
        <h2 class="year-title">${year} Papers</h2>
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

    const tagsHtml = weekData.tags?.map(tag => `<span><i class="fas fa-tag"></i> ${tag}</span>`).join('') || '';
    const dateHtml = weekData.date ? `<span><i class="far fa-calendar"></i> ${weekData.date}</span>` : '';
    const notebooksText = weekData.notebooks !== undefined ? `${weekData.notebooks} Notebook${weekData.notebooks !== 1 ? 's' : ''}` : 'Notebook';
    const exampleHtml = weekData.example ? `<p class="week-card-example"><strong>Пример:</strong> ${weekData.example}</p>` : '';

    card.innerHTML = `
        <div class="week-card-header">
            <h3 class="week-card-title">${weekData.title}</h3>
        </div>
        <div class="week-card-body">
            <div class="week-card-meta">${dateHtml} ${tagsHtml}</div>
            <p class="week-card-desc">${weekData.description || 'No description available.'}</p>
            <button class="btn btn-outline read-review">Read Review</button>
        </div>
        <div class="week-card-footer">
            <span><i class="far fa-file-alt"></i> ${weekData.papers} Paper${weekData.papers !== 1 ? 's' : ''}</span>
            <span><i class="far fa-file-code"></i> ${notebooksText}</span>
        </div>
    `;
    card.innerHTML = `
        <div class="week-card-header">
            <h3 class="week-card-title">${weekData.title}</h3>
        </div>
        <div class="week-card-body">
            <div class="week-card-meta">${dateHtml} ${tagsHtml}</div>
            <p class="week-card-desc">${weekData.description || 'No description available.'}</p>
            ${exampleHtml}
            <button class="btn btn-outline read-review">Read Review</button>
        </div>
        <div class="week-card-footer">
            <span><i class="far fa-file-alt"></i> ${weekData.papers} Paper${weekData.papers !== 1 ? 's' : ''}</span>
            <span><i class="far fa-file-code"></i> ${notebooksText}</span>
        </div>
    `;
    weeksGrid.appendChild(card);

    // Обработчик для кнопки чтения обзора
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

    loader.style.display = 'block';
    markdownContent.innerHTML = '';

    try {
        const response = await fetch(reviewUrl);
        if (!response.ok) throw new Error(`Failed to fetch review. Status: ${response.status}`);
        let markdown = await response.text();

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
        if (typeof marked === 'undefined') throw new Error("Marked.js library not loaded.");
        const html = marked.parse(markdown);

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
        if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
            MathJax.texReset?.();
            MathJax.typesetClear?.([markdownContent]);
            await MathJax.typesetPromise([markdownContent]);
        } else {
            console.warn("MathJax 3 not found or not configured.");
        }

        loader.style.display = 'none';
        return true;

    } catch (error) {
        console.error('Error loading or processing markdown:', error);
        markdownContent.innerHTML = `<div class="error-message"><h4>Error</h4><p>${error.message}</p></div>`;
        loader.style.display = 'none';
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

    modal.style.display = 'block';
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
            if (modal && (modal.style.display !== 'block' || !currentModalTitle || currentModalTitle.textContent !== title)) {
                 openReviewModal(year, week, title);
            }
        } else {
             // Хэш не соответствует формату, закрываем окно
             if (modal && modal.style.display === 'block') {
                 closeModal();
             }
        }
    } else {
        // Хэш пуст или не содержит '/', закрываем окно
        if (modal && modal.style.display === 'block') {
            closeModal();
        }
    }
}

function closeModal() {
    if (modal) {
        modal.style.display = 'none';
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
    if (event.key === 'Escape' && modal && modal.style.display === 'block') {
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

// --- Инициализация ---

window.addEventListener('DOMContentLoaded', loadWeeksData);
window.addEventListener('hashchange', checkUrlHash);