/**
 * Year Entity - Domain Model
 * Представляет сущность "Год исследований" в домене Research
 */
export class Year {
    constructor(year, weeks = []) {
        this.year = year;
        this.weeks = weeks;
    }

    /**
     * Добавляет неделю к году
     */
    addWeek(week) {
        this.weeks.push(week);
    }

    /**
     * Возвращает неделю по идентификатору
     */
    getWeek(weekId) {
        return this.weeks.find(week => week.getId() === weekId);
    }

    /**
     * Возвращает все недели
     */
    getWeeks() {
        return this.weeks;
    }

    /**
     * Возвращает количество недель
     */
    getWeeksCount() {
        return this.weeks.length;
    }

    /**
     * Возвращает общее количество статей за год
     */
    getTotalPapers() {
        return this.weeks.reduce((total, week) => total + (week.papers || 0), 0);
    }

    /**
     * Возвращает общее количество ноутбуков за год
     */
    getTotalNotebooks() {
        return this.weeks.reduce((total, week) => total + (week.notebooks || 0), 0);
    }

    /**
     * Возвращает все уникальные теги за год
     */
    getAllTags() {
        const allTags = this.weeks.flatMap(week => week.tags || []);
        return [...new Set(allTags)];
    }

    /**
     * Фильтрует недели по тегу
     */
    getWeeksByTag(tag) {
        return this.weeks.filter(week => week.hasTag(tag));
    }

    /**
     * Проверяет, есть ли недели в году
     */
    hasWeeks() {
        return this.weeks.length > 0;
    }

    /**
     * Возвращает последнюю добавленную неделю
     */
    getLatestWeek() {
        return this.weeks[this.weeks.length - 1];
    }

    /**
     * Возвращает недели, отсортированные по дате
     */
    getWeeksSortedByDate() {
        return [...this.weeks].sort((a, b) => {
            // Простая сортировка по номеру недели
            const weekNumA = parseInt(a.week.match(/\d+/)?.[0] || '0');
            const weekNumB = parseInt(b.week.match(/\d+/)?.[0] || '0');
            return weekNumA - weekNumB;
        });
    }

    /**
     * Возвращает статистику года
     */
    getStatistics() {
        return {
            year: this.year,
            weeksCount: this.getWeeksCount(),
            totalPapers: this.getTotalPapers(),
            totalNotebooks: this.getTotalNotebooks(),
            uniqueTags: this.getAllTags().length,
            tags: this.getAllTags()
        };
    }

    /**
     * Проверяет валидность года
     */
    isValid() {
        return this.year && this.year.toString().length === 4;
    }

    /**
     * Возвращает строковое представление года
     */
    toString() {
        return this.year.toString();
    }
} 