/**
 * LanguageSwitcher - Language Switch Component
 * Presentation Layer - Pixel-themed language switcher
 */
export class LanguageSwitcher {
    constructor(i18nService) {
        this.i18nService = i18nService;
        this.element = null;
        this.isOpen = false;

        this._initializeEventListeners();
    }

    /**
     * Create the language switcher element
     */
    createElement() {
        if (this.element) {
            return this.element;
        }

        const currentLang = this.i18nService.getCurrentLanguage();
        const supportedLanguages = this.i18nService.getSupportedLanguages();

        this.element = document.createElement('div');
        this.element.className = 'pixel-language-switcher';
        this.element.setAttribute('role', 'combobox');
        this.element.setAttribute('aria-label', this.i18nService.t('language.switch'));
        this.element.setAttribute('aria-expanded', 'false');

        this.element.innerHTML = `
            <button class="pixel-btn pixel-btn--sm pixel-language-switcher__trigger"
                    type="button"
                    aria-haspopup="listbox"
                    title="${this.i18nService.t('language.switch')}">
                <span class="language-flag">${this.i18nService.getLanguageFlag(currentLang)}</span>
                <span class="language-code">${currentLang.toUpperCase()}</span>
                <span class="dropdown-arrow">▼</span>
            </button>

            <div class="pixel-language-switcher__dropdown"
                 role="listbox"
                 aria-label="${this.i18nService.t('language.switch')}">
                <div class="pixel-card">
                    ${supportedLanguages.map(lang => `
                        <button class="pixel-language-switcher__option ${lang === currentLang ? 'active' : ''}"
                                type="button"
                                role="option"
                                data-language="${lang}"
                                aria-selected="${lang === currentLang}"
                                title="${this.i18nService.getLanguageDisplayName(lang)}">
                            <span class="language-flag">${this.i18nService.getLanguageFlag(lang)}</span>
                            <span class="language-name">${this.i18nService.getLanguageDisplayName(lang)}</span>
                            ${lang === currentLang ? '<span class="check-mark">✓</span>' : ''}
                        </button>
                    `).join('')}
                </div>
            </div>
        `;

        this._attachEventListeners();
        return this.element;
    }

    /**
     * Update the switcher when language changes
     */
    updateLanguage(newLanguage) {
        if (!this.element) return;

        const trigger = this.element.querySelector('.pixel-language-switcher__trigger');
        const flagElement = trigger.querySelector('.language-flag');
        const codeElement = trigger.querySelector('.language-code');

        // Update trigger button
        flagElement.textContent = this.i18nService.getLanguageFlag(newLanguage);
        codeElement.textContent = newLanguage.toUpperCase();
        trigger.title = this.i18nService.t('language.switch');

        // Update options
        const options = this.element.querySelectorAll('.pixel-language-switcher__option');
        options.forEach(option => {
            const lang = option.dataset.language;
            const isActive = lang === newLanguage;

            option.classList.toggle('active', isActive);
            option.setAttribute('aria-selected', isActive);

            // Update check mark
            const checkMark = option.querySelector('.check-mark');
            if (isActive && !checkMark) {
                option.innerHTML += '<span class="check-mark">✓</span>';
            } else if (!isActive && checkMark) {
                checkMark.remove();
            }
        });

        // Update ARIA labels
        this.element.setAttribute('aria-label', this.i18nService.t('language.switch'));
        const dropdown = this.element.querySelector('.pixel-language-switcher__dropdown');
        dropdown.setAttribute('aria-label', this.i18nService.t('language.switch'));
    }

    /**
     * Toggle dropdown visibility
     */
    toggle() {
        this.isOpen ? this.close() : this.open();
    }

    /**
     * Open dropdown
     */
    open() {
        if (!this.element || this.isOpen) return;

        this.isOpen = true;
        this.element.classList.add('open');
        this.element.setAttribute('aria-expanded', 'true');

        // Focus management
        const firstOption = this.element.querySelector('.pixel-language-switcher__option:not(.active)');
        if (firstOption) {
            firstOption.focus();
        }

        // Add global click listener to close dropdown
        setTimeout(() => {
            document.addEventListener('click', this._handleOutsideClick);
            document.addEventListener('keydown', this._handleKeydown);
        }, 0);
    }

    /**
     * Close dropdown
     */
    close() {
        if (!this.element || !this.isOpen) return;

        this.isOpen = false;
        this.element.classList.remove('open');
        this.element.setAttribute('aria-expanded', 'false');

        // Remove global listeners
        document.removeEventListener('click', this._handleOutsideClick);
        document.removeEventListener('keydown', this._handleKeydown);

        // Return focus to trigger
        const trigger = this.element.querySelector('.pixel-language-switcher__trigger');
        if (trigger) {
            trigger.focus();
        }
    }

    /**
     * Switch to specific language
     */
    async switchLanguage(language) {
        if (language === this.i18nService.getCurrentLanguage()) {
            this.close();
            return;
        }

        // Show loading state
        this._showLoadingState();

        try {
            const success = await this.i18nService.setLanguage(language);

            if (success) {
                this.updateLanguage(language);
                this._showSuccessAnimation();
                console.log(`🌐 Language switched to: ${language}`);
            } else {
                this._showErrorState();
                console.error(`❌ Failed to switch language to: ${language}`);
            }
        } catch (error) {
            this._showErrorState();
            console.error('❌ Language switch error:', error);
        } finally {
            this.close();
        }
    }

    /**
     * Initialize global event listeners
     */
    _initializeEventListeners() {
        // Bind methods to preserve context
        this._handleOutsideClick = this._handleOutsideClick.bind(this);
        this._handleKeydown = this._handleKeydown.bind(this);
        this._handleLanguageChange = this._handleLanguageChange.bind(this);

        // Listen for language changes from other sources
        document.addEventListener('languageChanged', this._handleLanguageChange);
    }

    /**
     * Attach event listeners to component elements
     */
    _attachEventListeners() {
        if (!this.element) return;

        const trigger = this.element.querySelector('.pixel-language-switcher__trigger');
        const options = this.element.querySelectorAll('.pixel-language-switcher__option');

        // Trigger button events
        trigger.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.toggle();
        });

        trigger.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.toggle();
            }
        });

        // Option button events
        options.forEach(option => {
            option.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const language = option.dataset.language;
                this.switchLanguage(language);
            });

            option.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    const language = option.dataset.language;
                    this.switchLanguage(language);
                }
            });
        });
    }

    /**
     * Handle clicks outside the component
     */
    _handleOutsideClick(event) {
        if (!this.element.contains(event.target)) {
            this.close();
        }
    }

    /**
     * Handle keyboard navigation
     */
    _handleKeydown(event) {
        if (!this.isOpen) return;

        const options = this.element.querySelectorAll('.pixel-language-switcher__option');
        const activeOption = document.activeElement;
        const currentIndex = Array.from(options).indexOf(activeOption);

        switch (event.key) {
            case 'Escape':
                event.preventDefault();
                this.close();
                break;

            case 'ArrowDown':
                event.preventDefault();
                const nextIndex = (currentIndex + 1) % options.length;
                options[nextIndex].focus();
                break;

            case 'ArrowUp':
                event.preventDefault();
                const prevIndex = currentIndex > 0 ? currentIndex - 1 : options.length - 1;
                options[prevIndex].focus();
                break;

            case 'Home':
                event.preventDefault();
                options[0].focus();
                break;

            case 'End':
                event.preventDefault();
                options[options.length - 1].focus();
                break;
        }
    }

    /**
     * Handle language change events from i18n service
     */
    _handleLanguageChange(event) {
        const { currentLanguage } = event.detail;
        this.updateLanguage(currentLanguage);
    }

    /**
     * Show loading state during language switch
     */
    _showLoadingState() {
        if (!this.element) return;

        const trigger = this.element.querySelector('.pixel-language-switcher__trigger');
        trigger.classList.add('loading');
        trigger.disabled = true;

        // Add loading animation
        const arrow = trigger.querySelector('.dropdown-arrow');
        if (arrow) {
            arrow.textContent = '⟳';
            arrow.style.animation = 'spin 1s linear infinite';
        }
    }

    /**
     * Show success animation after language switch
     */
    _showSuccessAnimation() {
        if (!this.element) return;

        const trigger = this.element.querySelector('.pixel-language-switcher__trigger');
        trigger.classList.remove('loading');
        trigger.classList.add('success');
        trigger.disabled = false;

        // Reset arrow
        const arrow = trigger.querySelector('.dropdown-arrow');
        if (arrow) {
            arrow.textContent = '▼';
            arrow.style.animation = '';
        }

        // Remove success class after animation
        setTimeout(() => {
            trigger.classList.remove('success');
        }, 1000);
    }

    /**
     * Show error state
     */
    _showErrorState() {
        if (!this.element) return;

        const trigger = this.element.querySelector('.pixel-language-switcher__trigger');
        trigger.classList.remove('loading');
        trigger.classList.add('error');
        trigger.disabled = false;

        // Reset arrow
        const arrow = trigger.querySelector('.dropdown-arrow');
        if (arrow) {
            arrow.textContent = '▼';
            arrow.style.animation = '';
        }

        // Remove error class after animation
        setTimeout(() => {
            trigger.classList.remove('error');
        }, 2000);
    }

    /**
     * Destroy the component and clean up
     */
    destroy() {
        if (this.element) {
            this.close();
            this.element.remove();
            this.element = null;
        }

        // Remove global event listeners
        document.removeEventListener('languageChanged', this._handleLanguageChange);
        document.removeEventListener('click', this._handleOutsideClick);
        document.removeEventListener('keydown', this._handleKeydown);
    }

    /**
     * Get current state for debugging
     */
    getState() {
        return {
            isOpen: this.isOpen,
            currentLanguage: this.i18nService.getCurrentLanguage(),
            supportedLanguages: this.i18nService.getSupportedLanguages(),
            elementExists: !!this.element
        };
    }
}