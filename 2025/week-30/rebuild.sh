#!/bin/bash
echo "🔄 Пересборка Jupyter Book..."
jb build .
echo "✅ Сборка завершена!"
echo "🌐 Открытие в браузере..."
open _build/html/index.html
echo "🎉 Готово!"
