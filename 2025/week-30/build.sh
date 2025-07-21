#!/bin/bash
echo "🔄 Пересборка Jupyter Book..."
/Users/me/Library/Python/3.9/bin/jupyter-book build .
echo "✅ Сборка завершена!"
echo "🌐 Открытие в браузере..."
open _build/html/index.html
echo "🎉 Готово!"
