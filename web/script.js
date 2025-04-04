// script.js

// Конфигурация для GitHub API
const GITHUB_REPO = 'Verbasik/Weekly-arXiv-ML-AI-Research-Review';
const GITHUB_BRANCH = 'master';

// Функция для загрузки данных из index.json
async function loadWeeksData() {
    const jsonUrl = `https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/web/index.json`;
    const content = document.querySelector('.content'); // Кешируем элемент

    try {
        const response = await fetch(jsonUrl);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Очистить существующие секции с годами (кроме #home)
        document.querySelectorAll('.year-section').forEach(section => {
            if (section.id !== 'home') {
                section.remove();
            }
        });

        // Создать секции для каждого года и заполнить их карточками
        for (const yearData of data.years) {
            const year = yearData.year;
            const yearSection = createYearSection(year, content); // Передаем content

            for (const weekData of yearData.weeks) {
                createWeekCard(yearSection, year, weekData);
            }
        }

        // Обновить фильтры в боковой панели
        updateYearFilters(data.years.map(y => y.year));

        // Проверить хэш URL после загрузки данных
        checkUrlHash();

    } catch (error) {
        console.error('Error loading weeks data:', error);
        // Показать сообщение об ошибке на странице
        if (content) {
            content.innerHTML += `
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Could not load data from the repository. Please check your connection or try again later.</p>
                    <p>Error: ${error.message}</p>
                </div>
            `;
        }
    }
}

// Функция для создания секции года
function createYearSection(year, contentElement) {
    const yearSection = document.createElement('section');
    yearSection.id = year;
    yearSection.className = 'year-section';

    yearSection.innerHTML = `
        <h2 class="year-title">${year} Papers</h2>
        <div class="weeks-grid"></div>
    `;

    if (contentElement) {
        contentElement.appendChild(yearSection);
    }
    return yearSection;
}

// Функция для создания карточки недели
function createWeekCard(yearSection, year, weekData) {
    const weeksGrid = yearSection.querySelector('.weeks-grid');
    if (!weeksGrid) return; // Добавлена проверка

    const card = document.createElement('div');
    card.className = 'week-card';
    card.setAttribute('data-week', weekData.week);
    card.setAttribute('data-year', year);

    // Формирование тегов
    const tagsHtml = weekData.tags && weekData.tags.length ?
        weekData.tags.map(tag => `<span><i class="fas fa-tag"></i> ${tag}</span>`).join('') :
        '';

    // Формирование даты
    const dateHtml = weekData.date ?
        `<span><i class="far fa-calendar"></i> ${weekData.date}</span>` :
        '';

    card.innerHTML = `
        <div class="week-card-header">
            <h3 class="week-card-title">${weekData.title}</h3>
        </div>
        <div class="week-card-body">
            <div class="week-card-meta">
                ${dateHtml}
                ${tagsHtml}
            </div>
            <p class="week-card-desc">${weekData.description || 'No description available.'}</p>
            <button class="btn btn-outline read-review">Read Review</button>
        </div>
        <div class="week-card-footer">
            <span><i class="far fa-file-alt"></i> ${weekData.papers} Paper${weekData.papers !== 1 ? 's' : ''}</span>
            <span><i class="far fa-file-code"></i> ${weekData.notebooks !== undefined ? `${weekData.notebooks} Notebook${weekData.notebooks !== 1 ? 's' : ''}` : 'Notebook'}</span>
        </div>
    `;

    weeksGrid.appendChild(card);

    // Добавить обработчик события для кнопки
    const readReviewButton = card.querySelector('.read-review');
    if (readReviewButton) {
        readReviewButton.addEventListener('click', async (e) => {
            e.preventDefault();
            openReviewModal(year, weekData.week, weekData.title);
        });
    }
}

// Функция для открытия модального окна с обзором
async function openReviewModal(year, week, title) {
    const modal = document.getElementById('markdown-modal');
    const markdownContent = document.getElementById('markdown-content');
    if (!modal || !markdownContent) return;

    // Set modal title dynamically
    const existingTitle = modal.querySelector('.modal-content h2');
    if (existingTitle) existingTitle.remove(); // Remove old title if exists

    const titleElement = document.createElement('h2');
    titleElement.textContent = title;
    titleElement.style.marginTop = '0'; // Add some style if needed
    titleElement.style.marginBottom = '1rem';
    markdownContent.before(titleElement); // Insert title before content area

    // Show modal
    modal.style.display = 'block';

    // Load the markdown content
    await loadMarkdownFromGitHub(year, week);

    // Update URL hash
    window.location.hash = `#${year}/${week}`;
}


// Функция для обновления фильтров годов в боковой панели
function updateYearFilters(years) {
    const yearFilterList = document.querySelector('.sidebar ul:first-of-type');
    if (!yearFilterList) return; // Добавлена проверка

    yearFilterList.innerHTML = ''; // Очищаем список

    years.sort((a, b) => b - a); // Сортируем года по убыванию

    years.forEach((year, index) => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = `#${year}`;
        a.textContent = year;
        if (index === 0) { // Делаем первый (самый новый) год активным
            a.className = 'active';
        }
        li.appendChild(a);
        yearFilterList.appendChild(li);
    });

    // Обновить обработчики событий для фильтров
    document.querySelectorAll('.sidebar a[href^="#"]').forEach(link => {
        // Удаляем старые обработчики, если они были
        link.replaceWith(link.cloneNode(true));
    });
    // Добавляем новые обработчики
    document.querySelectorAll('.sidebar a[href^="#"]').forEach(link => {
        link.addEventListener('click', function(e) {
            // Убрать класс active со всех ссылок в этом списке
            yearFilterList.querySelectorAll('a').forEach(a => a.classList.remove('active'));
            // Добавить класс active к нажатой ссылке
            this.classList.add('active');

            // Плавная прокрутка к секции года (если она есть)
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                // e.preventDefault(); // Предотвращаем стандартный переход по якорю, если нужна плавная прокрутка
                // targetElement.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

// Get modal elements (лучше делать это один раз)
const modal = document.getElementById('markdown-modal');
const closeModal = document.querySelector('.close-modal');
const markdownContent = document.getElementById('markdown-content');
const loader = document.querySelector('.loader');

// Function to load markdown content from GitHub
async function loadMarkdownFromGitHub(year, week) {
    const reviewUrl = `https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/${year}/${week}/review.md`;

    if (!loader || !markdownContent) return false; // Добавлена проверка

    // Show loading spinner
    loader.style.display = 'block';
    markdownContent.innerHTML = ''; // Очищаем предыдущее содержимое

    try {
        const response = await fetch(reviewUrl);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const markdown = await response.text();

        // Render markdown to HTML using Marked.js (убедитесь, что Marked.js загружен)
        if (typeof marked !== 'undefined') {
            const html = marked.parse(markdown);
            markdownContent.innerHTML = html;
        } else {
            console.error("Marked.js library is not loaded.");
            markdownContent.innerHTML = '<p>Error: Markdown library not loaded.</p>';
        }


        // Process LaTeX in the loaded content using MathJax (убедитесь, что MathJax загружен и сконфигурирован)
        if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
            // Очищаем предыдущие обработанные элементы MathJax, если необходимо
            MathJax.texReset && MathJax.texReset();
            MathJax.typesetClear && MathJax.typesetClear([markdownContent]);

            await MathJax.typesetPromise([markdownContent]);
        } else if (typeof MathJax !== 'undefined' && MathJax.Hub) {
             // Для MathJax v2 (если используется)
             MathJax.Hub.Queue(["Typeset", MathJax.Hub, markdownContent]);
        } else {
            console.warn("MathJax library is not loaded or configured properly.");
        }

        loader.style.display = 'none'; // Hide loader after processing
        return true;

    } catch (error) {
        console.error('Error loading markdown:', error);
        markdownContent.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Could not load the review file. Please check if it exists in the repository.</p>
                <p>Path: ${year}/${week}/review.md</p>
                <p>Error: ${error.message}</p>
            </div>
        `;
        loader.style.display = 'none';
        return false;
    }
}

// Event listeners (добавляем после определения функций)

// Close modal when clicking on X
if (closeModal) {
    closeModal.addEventListener('click', () => {
        if (modal) modal.style.display = 'none';
    });
}

// Close modal when clicking outside of it
window.addEventListener('click', (e) => {
    if (e.target === modal) {
        modal.style.display = 'none';
    }
});

// Tab switching functionality (если используется)
document.querySelectorAll('.tab')?.forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs and content in the same group
        const tabGroup = tab.closest('.tabs'); // Находим родительский контейнер табов
        if (!tabGroup) return;

        tabGroup.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        const contentContainer = tabGroup.nextElementSibling; // Предполагаем, что контент идет сразу после табов
        if (contentContainer) {
            contentContainer.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        }

        // Add active class to clicked tab and corresponding content
        tab.classList.add('active');
        const tabId = tab.getAttribute('data-tab');
        const targetContent = document.getElementById(`${tabId}-tab`);
        if (targetContent) {
            targetContent.classList.add('active');
        }
    });
});

// Back to top button functionality
const backToTopButton = document.getElementById('back-to-top');
if (backToTopButton) {
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            backToTopButton.classList.add('visible');
        } else {
            backToTopButton.classList.remove('visible');
        }
    });

    backToTopButton.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

// Search functionality
const searchInput = document.querySelector('.search-bar input');
const searchButton = document.querySelector('.search-bar button');

function performSearch(query) {
    alert(`Search functionality is not implemented yet. You searched for: ${query}`);
    // Здесь должна быть логика поиска/фильтрации карточек
    // Например, можно пройтись по всем .week-card и скрыть те, что не соответствуют запросу
}

if (searchButton && searchInput) {
    searchButton.addEventListener('click', () => {
        performSearch(searchInput.value.trim());
    });

    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch(searchInput.value.trim());
        }
    });
}

// Check URL hash on page load and when hash changes
function checkUrlHash() {
    const hash = window.location.hash;

    if (hash && hash.includes('/')) {
        // Format: #YYYY/week-XX
        const parts = hash.substring(1).split('/');
        if (parts.length === 2) {
            const year = parts[0];
            const week = parts[1];

            // Find the corresponding card
            const card = document.querySelector(`.week-card[data-year="${year}"][data-week="${week}"]`);

            if (card) {
                const title = card.querySelector('.week-card-title')?.textContent || 'Review';
                // Открываем модальное окно
                openReviewModal(year, week, title);

                // Прокручиваем до карточки (опционально)
                // card.scrollIntoView({ behavior: 'smooth', block: 'center' });
            } else {
                 console.warn(`Card not found for hash: ${hash}`);
                 // Можно закрыть модальное окно, если оно было открыто
                 if (modal) modal.style.display = 'none';
            }
        }
    } else {
        // Если хэш не соответствует формату или пуст, закрываем модальное окно
        if (modal) modal.style.display = 'none';
    }
}

// Initial load
window.addEventListener('DOMContentLoaded', () => {
    loadWeeksData(); // Загружаем данные после загрузки DOM
    // checkUrlHash(); // Проверяем хэш после загрузки данных (перенесено в loadWeeksData)
});

// Listen for hash changes to open/close modal
window.addEventListener('hashchange', checkUrlHash);