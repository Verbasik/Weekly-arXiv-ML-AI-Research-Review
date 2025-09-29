/**
 * I18nService - Internationalization Service
 * Infrastructure Layer - Manages translations and language switching
 */
export class I18nService {
    constructor() {
        this.currentLanguage = 'ru'; // Default to Russian
        this.translations = new Map();
        this.fallbackLanguage = 'ru';
        this.supportedLanguages = ['ru', 'en'];

        // Browser language detection
        this.browserLanguage = this._detectBrowserLanguage();

        // Load initial language from localStorage or browser
        this.currentLanguage = this._getInitialLanguage();

        console.log(`🌐 I18nService initialized with language: ${this.currentLanguage}`);
    }

    /**
     * Initialize the service and load translations
     */
    async initialize() {
        try {
            await this.loadTranslations(this.currentLanguage);
            this._saveLanguagePreference();
            return true;
        } catch (error) {
            console.error('❌ Failed to initialize I18nService:', error);
            // Fallback to Russian if initialization fails
            if (this.currentLanguage !== this.fallbackLanguage) {
                await this.loadTranslations(this.fallbackLanguage);
                this.currentLanguage = this.fallbackLanguage;
            }
            return false;
        }
    }

    /**
     * Get current language
     */
    getCurrentLanguage() {
        return this.currentLanguage;
    }

    /**
     * Get all supported languages
     */
    getSupportedLanguages() {
        return [...this.supportedLanguages];
    }

    /**
     * Check if language is supported
     */
    isLanguageSupported(language) {
        return this.supportedLanguages.includes(language);
    }

    /**
     * Change current language
     */
    async setLanguage(language) {
        if (!this.isLanguageSupported(language)) {
            console.warn(`⚠️ Language '${language}' is not supported`);
            return false;
        }

        if (language === this.currentLanguage) {
            return true; // Already set
        }

        try {
            // Load translations if not cached
            if (!this.translations.has(language)) {
                await this.loadTranslations(language);
            }

            const previousLanguage = this.currentLanguage;
            this.currentLanguage = language;

            // Save preference
            this._saveLanguagePreference();

            // Emit language change event
            this._emitLanguageChangeEvent(language, previousLanguage);

            console.log(`🌐 Language changed to: ${language}`);
            return true;
        } catch (error) {
            console.error(`❌ Failed to change language to '${language}':`, error);
            return false;
        }
    }

    /**
     * Get translation for key
     */
    t(key, params = {}) {
        const translation = this._getTranslation(key, this.currentLanguage);
        return this._interpolateParams(translation, params);
    }

    /**
     * Get translation with explicit language
     */
    tl(key, language, params = {}) {
        const translation = this._getTranslation(key, language);
        return this._interpolateParams(translation, params);
    }

    /**
     * Check if translation exists
     */
    hasTranslation(key, language = this.currentLanguage) {
        return this._getTranslation(key, language, false) !== null;
    }

    /**
     * Load translations for a language
     */
    async loadTranslations(language) {
        if (this.translations.has(language)) {
            return this.translations.get(language);
        }

        const url = `web/infrastructure/i18n/locales/${language}.json`;

        try {
            console.log(`📥 Loading translations for: ${language}`);
            const response = await fetch(url);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const translations = await response.json();
            this.translations.set(language, translations);

            console.log(`✅ Translations loaded for: ${language}`);
            return translations;
        } catch (error) {
            console.error(`❌ Failed to load translations for '${language}':`, error);
            throw error;
        }
    }

    /**
     * Get translation from cache
     */
    _getTranslation(key, language, useFallback = true) {
        const translations = this.translations.get(language);

        if (translations) {
            const value = this._getNestedProperty(translations, key);
            if (value !== undefined && value !== null) {
                return value;
            }
        }

        // Try fallback language
        if (useFallback && language !== this.fallbackLanguage) {
            const fallbackTranslations = this.translations.get(this.fallbackLanguage);
            if (fallbackTranslations) {
                const fallbackValue = this._getNestedProperty(fallbackTranslations, key);
                if (fallbackValue !== undefined && fallbackValue !== null) {
                    console.warn(`🔄 Using fallback translation for key: ${key}`);
                    return fallbackValue;
                }
            }
        }

        // Return key as fallback
        console.warn(`❌ Missing translation for key: ${key} (language: ${language})`);
        return key;
    }

    /**
     * Get nested property from object using dot notation
     */
    _getNestedProperty(obj, path) {
        return path.split('.').reduce((current, key) => {
            return current && current[key] !== undefined ? current[key] : undefined;
        }, obj);
    }

    /**
     * Interpolate parameters in translation string
     */
    _interpolateParams(text, params) {
        if (!params || Object.keys(params).length === 0) {
            return text;
        }

        return text.replace(/\{\{(\w+)\}\}/g, (match, key) => {
            return params[key] !== undefined ? params[key] : match;
        });
    }

    /**
     * Detect browser language
     */
    _detectBrowserLanguage() {
        const language = navigator.language || navigator.userLanguage || 'ru';
        const shortLang = language.split('-')[0]; // 'en-US' -> 'en'
        return this.isLanguageSupported(shortLang) ? shortLang : 'ru';
    }

    /**
     * Get initial language from storage or browser
     */
    _getInitialLanguage() {
        try {
            const savedLanguage = localStorage.getItem('twrb_language');
            if (savedLanguage && this.isLanguageSupported(savedLanguage)) {
                return savedLanguage;
            }
        } catch (error) {
            console.warn('⚠️ Could not access localStorage for language preference');
        }

        return this.browserLanguage;
    }

    /**
     * Save language preference to localStorage
     */
    _saveLanguagePreference() {
        try {
            localStorage.setItem('twrb_language', this.currentLanguage);
        } catch (error) {
            console.warn('⚠️ Could not save language preference to localStorage');
        }
    }

    /**
     * Emit language change event
     */
    _emitLanguageChangeEvent(newLanguage, previousLanguage) {
        const event = new CustomEvent('languageChanged', {
            detail: {
                currentLanguage: newLanguage,
                previousLanguage: previousLanguage,
                supportedLanguages: this.supportedLanguages,
                i18nService: this
            }
        });

        document.dispatchEvent(event);
    }

    /**
     * Get language display name
     */
    getLanguageDisplayName(language = this.currentLanguage) {
        const displayNames = {
            'ru': 'Русский',
            'en': 'English'
        };
        return displayNames[language] || language;
    }

    /**
     * Get language flag emoji
     */
    getLanguageFlag(language = this.currentLanguage) {
        const flags = {
            'ru': '🇷🇺',
            'en': '🇺🇸'
        };
        return flags[language] || '🌐';
    }

    /**
     * Get text direction for language
     */
    getTextDirection(language = this.currentLanguage) {
        const rtlLanguages = ['ar', 'he', 'fa']; // Future RTL support
        return rtlLanguages.includes(language) ? 'rtl' : 'ltr';
    }

    /**
     * Format date according to current language
     */
    formatDate(date, options = {}) {
        const locale = this.currentLanguage === 'ru' ? 'ru-RU' : 'en-US';
        return new Intl.DateTimeFormat(locale, options).format(date);
    }

    /**
     * Format number according to current language
     */
    formatNumber(number, options = {}) {
        const locale = this.currentLanguage === 'ru' ? 'ru-RU' : 'en-US';
        return new Intl.NumberFormat(locale, options).format(number);
    }

    /**
     * Preload translations for better performance
     */
    async preloadAllTranslations() {
        const promises = this.supportedLanguages.map(lang => {
            if (!this.translations.has(lang)) {
                return this.loadTranslations(lang).catch(error => {
                    console.warn(`⚠️ Failed to preload ${lang} translations:`, error);
                });
            }
            return Promise.resolve();
        });

        await Promise.all(promises);
        console.log('✅ All translations preloaded');
    }

    /**
     * Clear translations cache
     */
    clearCache() {
        this.translations.clear();
        console.log('🗑️ Translations cache cleared');
    }

    /**
     * Get statistics about loaded translations
     */
    getStats() {
        const stats = {
            currentLanguage: this.currentLanguage,
            supportedLanguages: this.supportedLanguages,
            loadedLanguages: Array.from(this.translations.keys()),
            cacheSize: this.translations.size
        };

        // Count translation keys per language
        stats.translationCounts = {};
        for (const [lang, translations] of this.translations) {
            stats.translationCounts[lang] = this._countKeys(translations);
        }

        return stats;
    }

    /**
     * Recursively count keys in translation object
     */
    _countKeys(obj) {
        let count = 0;
        for (const key in obj) {
            if (typeof obj[key] === 'object' && obj[key] !== null) {
                count += this._countKeys(obj[key]);
            } else {
                count++;
            }
        }
        return count;
    }
}