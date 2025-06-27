import { Project } from '../entities/Project.js';

/**
 * Agents Repository - Data Access Layer
 * Абстракция для работы с данными agents проектов
 */
export class AgentsRepository {
    constructor(dataSource) {
        this.dataSource = dataSource;
    }

    /**
     * Получает все проекты
     */
    async getAllProjects() {
        try {
            const data = await this.dataSource.fetchData();
            return this._mapToProjectEntities(data.projects);
        } catch (error) {
            throw new Error(`Failed to fetch agents data: ${error.message}`);
        }
    }

    /**
     * Получает конкретный проект
     */
    async getProject(projectId) {
        const projects = await this.getAllProjects();
        return projects.find(project => project.getId() === projectId);
    }

    /**
     * Получает все доступные теги
     */
    async getAllTags() {
        const projects = await this.getAllProjects();
        const allTags = projects.flatMap(project => project.tags);
        return [...new Set(allTags)];
    }

    /**
     * Фильтрует проекты по тегу
     */
    async getProjectsByTag(tag) {
        const projects = await this.getAllProjects();
        return projects.filter(project => project.hasTag(tag));
    }

    /**
     * Получает статистику по всем проектам
     */
    async getStatistics() {
        const projects = await this.getAllProjects();
        return {
            totalProjects: projects.length,
            totalNotebooks: projects.reduce((total, project) => total + (project.notebooks || 0), 0),
            totalCode: projects.reduce((total, project) => total + (project.code || 0), 0),
            totalDemos: projects.reduce((total, project) => total + (project.demo || 0), 0),
            projectStatistics: projects.map(project => ({
                id: project.getId(),
                title: project.title,
                tags: project.tags,
                difficulty: project.difficulty,
                technologies: project.technologies
            }))
        };
    }

    /**
     * Получает контент markdown для проекта
     */
    async getProjectMarkdown(projectId) {
        try {
            return await this.dataSource.fetchMarkdown(projectId);
        } catch (error) {
            throw new Error(`Failed to fetch markdown for ${projectId}: ${error.message}`);
        }
    }

    /**
     * Поиск проектов по запросу
     */
    async searchProjects(query) {
        const projects = await this.getAllProjects();
        const results = [];
        
        const searchTerm = query.toLowerCase();
        
        projects.forEach(project => {
            const matchesTitle = project.title.toLowerCase().includes(searchTerm);
            const matchesDescription = project.description.toLowerCase().includes(searchTerm);
            const matchesTags = project.tags.some(tag => tag.toLowerCase().includes(searchTerm));
            const matchesTechnologies = project.technologies.some(tech => tech.toLowerCase().includes(searchTerm));
            
            if (matchesTitle || matchesDescription || matchesTags || matchesTechnologies) {
                results.push({
                    project: project,
                    relevance: this._calculateRelevance(project, searchTerm)
                });
            }
        });
        
        // Сортируем по релевантности
        return results.sort((a, b) => b.relevance - a.relevance);
    }

    /**
     * Фильтрует проекты по сложности
     */
    async getProjectsByDifficulty(difficulty) {
        const projects = await this.getAllProjects();
        return projects.filter(project => project.difficulty === difficulty);
    }

    /**
     * Получает проекты по технологии
     */
    async getProjectsByTechnology(technology) {
        const projects = await this.getAllProjects();
        return projects.filter(project => 
            project.technologies.some(tech => 
                tech.toLowerCase().includes(technology.toLowerCase())
            )
        );
    }

    /**
     * Преобразует сырые данные в сущности Project
     */
    _mapToProjectEntities(projectsData) {
        return projectsData.map(projectData => new Project(projectData));
    }

    /**
     * Вычисляет релевантность поиска
     */
    _calculateRelevance(project, searchTerm) {
        let relevance = 0;
        
        // Вес для совпадения в заголовке
        if (project.title.toLowerCase().includes(searchTerm)) {
            relevance += 10;
        }
        
        // Вес для совпадения в описании
        if (project.description.toLowerCase().includes(searchTerm)) {
            relevance += 5;
        }
        
        // Вес для совпадения в тегах
        project.tags.forEach(tag => {
            if (tag.toLowerCase().includes(searchTerm)) {
                relevance += 7;
            }
        });

        // Вес для совпадения в технологиях
        project.technologies.forEach(tech => {
            if (tech.toLowerCase().includes(searchTerm)) {
                relevance += 8;
            }
        });
        
        return relevance;
    }
} 