"""
Красивый интерфейс для анализа энтропии токенов модели Gemma-3n-E2B-it
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Добавляем путь к модулю
sys.path.append(str(Path(__file__).parent))

from model_loader import GemmaEntropyAnalyzer

class EntropyExperimentRunner:
    """Класс для красивого запуска и отображения экспериментов с энтропией"""
    
    def __init__(self):
        self.analyzer = None
        
    def print_header(self):
        """Красивый заголовок эксперимента"""
        print("\n" + "="*80)
        print("🧠 ЭКСПЕРИМЕНТ: Анализ энтропии токенов модели Gemma-3n-E2B-it")
        print("="*80)
        print("📊 Цель: Измерить неопределенность модели в предсказании токенов")
        print("📐 Формула: H_i = -∑P_i(j) * log₂(P_i(j))")
        print("="*80 + "\n")
        
    def print_section(self, title: str, emoji: str = "📋"):
        """Печать заголовка секции"""
        print(f"\n{emoji} {title}")
        print("-" * (len(title) + 4))
        
    def print_model_info(self, info: Dict[str, Any]):
        """Красивый вывод информации о модели"""
        self.print_section("ИНФОРМАЦИЯ О МОДЕЛИ", "🤖")
        
        print(f"📛 Модель: {info['model_name']}")
        print(f"📚 Размер словаря: {info['vocab_size']:,} токенов")
        print(f"💻 Устройство: {info['device']}")
        print(f"🔢 Параметры: {info['parameters']:,}")
        print(f"🎯 Тип: {info['model_type']}")
        
    def print_generation_analysis(self, prompt: str, results: Dict[str, Any]):
        """Красивый вывод генеративного анализа энтропии"""
        
        self.print_section(f"ГЕНЕРАЦИЯ С АНАЛИЗОМ ЭНТРОПИИ", "🎲")
        print(f"🎯 Промпт: '{prompt}'")
        print(f"✨ Сгенерировано: '{results['full_generated_text']}'")
        print(f"📝 Полный текст: '{results['complete_text']}'")
        
        generated_parts = results['generated_text_parts']
        entropies = results['entropies']
        probabilities = results['probabilities']
        
        print(f"\nПроцесс генерации ({len(generated_parts)} токенов):")
        print("=" * 70)
        
        cumulative_text = prompt
        for i, (token_part, entropy, prob) in enumerate(zip(generated_parts, entropies, probabilities)):
            # Очищаем токен для отображения
            clean_token = token_part.replace('▁', ' ')
                
            # Интерпретация энтропии
            if entropy < 0.1:
                interpretation = "Очень уверен"
            elif entropy < 0.5:
                interpretation = "Уверен"  
            elif entropy < 1.0:
                interpretation = "Средняя неопределенность"
            elif entropy < 2.0:
                interpretation = "Неуверен"
            else:
                interpretation = "Очень неуверен"
            
            print(f"Шаг {i+1:2d}: Токен '{clean_token}'")
            print(f"         Энтропия: {entropy:6.3f} бит")
            print(f"         Вероятность: {prob:6.3f}")
            print(f"         Уверенность: {interpretation}")
            print("-" * 50)
            cumulative_text += token_part
        
        # Статистика генерации
        if entropies:
            mean_entropy = sum(entropies) / len(entropies)
            min_entropy = min(entropies)
            max_entropy = max(entropies)
            
            self.print_section("СТАТИСТИКА ГЕНЕРАЦИИ", "📈")
            print(f"📊 Средняя энтропия генерации: {mean_entropy:.3f} бит")
            print(f"📉 Минимальная энтропия:       {min_entropy:.3f} бит")
            print(f"📈 Максимальная энтропия:      {max_entropy:.3f} бит")
            print(f"🎲 Количество шагов:           {len(entropies)}")
            
            # Анализ процесса генерации
            if mean_entropy < 0.5:
                generation_analysis = "Модель генерировала очень уверенно"
                emoji = "🎯"
            elif mean_entropy < 1.0:
                generation_analysis = "Уверенная генерация"
                emoji = "✅"
            elif mean_entropy < 1.5:
                generation_analysis = "Умеренная неопределенность в генерации"
                emoji = "🤔"
            else:
                generation_analysis = "Высокая неопределенность в генерации"
                emoji = "❓"
                
            print(f"\n{emoji} Анализ генерации: {generation_analysis}")
            
            # Найдем самый уверенный и неуверенный шаги
            min_idx = entropies.index(min_entropy)
            max_idx = entropies.index(max_entropy)
            
            print(f"🎯 Самый уверенный шаг: '{generated_parts[min_idx]}' (энтропия: {min_entropy:.3f})")
            print(f"❓ Самый неуверенный шаг: '{generated_parts[max_idx]}' (энтропия: {max_entropy:.3f})")

    def print_entropy_analysis(self, text: str, results: Dict[str, Any]):
        """Красивый вывод анализа энтропии (для существующего текста)"""
        tokens = results['tokens']
        entropies = results['entropy'][0]  # Убираем batch dimension
        
        self.print_section(f"АНАЛИЗ ВХОДНОГО ТЕКСТА: '{text}'", "🔍")
        
        print("Анализ токенов:")
        print("=" * 50)
            
        for i, (token, entropy) in enumerate(zip(tokens, entropies)):
            # Очищаем токен для красивого отображения
            clean_token = token.replace('▁', ' ').replace('<bos>', '[НАЧАЛО]')
            
            # Интерпретация уровня энтропии
            entropy_val = float(entropy)
            if entropy_val < 0.1:
                interpretation = "Очень уверен"
            elif entropy_val < 0.5:
                interpretation = "Уверен"
            elif entropy_val < 1.0:
                interpretation = "Средняя неопределенность"
            elif entropy_val < 2.0:
                interpretation = "Неуверен"
            else:
                interpretation = "Очень неуверен"
                
            print(f"Токен {i+1:2d}: '{clean_token}'")
            print(f"          Энтропия: {entropy_val:6.3f} бит")
            print(f"          Уверенность: {interpretation}")
            print("-" * 40)
        
    def run_experiment(self, test_texts: List[str]):
        """Запуск полного эксперимента"""
        self.print_header()
        
        # Инициализация
        self.print_section("ИНИЦИАЛИЗАЦИЯ", "🚀")
        print("⏳ Загружаем анализатор энтропии...")
        
        try:
            self.analyzer = GemmaEntropyAnalyzer()
            print("✅ Анализатор создан")
            
            print("⏳ Загружаем модель Gemma-3n-E2B-it...")
            print("   (Это может занять несколько минут при первой загрузке)")
            
            start_time = time.time()
            self.analyzer.load_model()
            load_time = time.time() - start_time
            
            print(f"✅ Модель загружена за {load_time:.1f} секунд")
            
            # Информация о модели
            model_info = self.analyzer.get_model_info()
            self.print_model_info(model_info)
            
            # Генеративный анализ энтропии
            self.print_section("ГЕНЕРАТИВНЫЙ АНАЛИЗ ЭНТРОПИИ", "🎲")
            
            generation_prompts = [
                "Теорема Пифагора гласит, что",
                "2 + 2 =",
                "В далекой галактике",
                "Столица России"
            ]
            
            for i, prompt in enumerate(generation_prompts, 1):
                print(f"\n🎯 Генерация {i}/{len(generation_prompts)}")
                print("⏳ Генерируем текст с анализом энтропии...")
                
                start_time = time.time()
                results = self.analyzer.generate_with_entropy_analysis(prompt, max_new_tokens=8)
                generation_time = time.time() - start_time
                
                self.print_generation_analysis(prompt, results)
                print(f"⏱️  Время генерации: {generation_time:.2f} секунд")
                
                if i < len(generation_prompts):
                    print("\n" + "─"*80)
                    
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False
            
        self.print_section("ЭКСПЕРИМЕНТ ЗАВЕРШЕН", "🎉")
        print("✅ Все тесты выполнены успешно!")
        print("📊 Результаты показывают различные уровни неопределенности модели")
        print("🔬 Энтропия помогает понять, насколько модель уверена в своих предсказаниях")
        print("\n" + "="*80 + "\n")
        
        return True

def main():
    """Основная функция для запуска эксперимента"""
    
    # Тестовые тексты разной сложности
    test_texts = [
        "Привет, мир!",
        "Теорема Пифагора гласит, что",
        "В далекой галактике",
        "2 + 2 =",
        "Что вы думаете о"
    ]
    
    runner = EntropyExperimentRunner()
    success = runner.run_experiment(test_texts)
    
    if not success:
        print("💥 Эксперимент завершился с ошибкой")
        sys.exit(1)

if __name__ == "__main__":
    main()