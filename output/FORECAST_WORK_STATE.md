# Прогноз завершён

18.10.2026. Результаты: output/joint_forecast/forecast_report.html, final_project.ipynb, submission_bundle.zip. Все 16 целей, M1–M4 и POOL, 6 validation-месяцев и 21 test-месяц. M3 выбрана заранее, q=.015 для M4. Daily scaled CRPS улучшен на 10,1767%, RMSE на 1,7044%. Важное ограничение: monthly coverage M3 72,1%/90,0%, M4 77,9%/91,8%; явно раскрыто в HTML и notebook.

22 теста PASS; verification PASS; 290 component fits, Rhat<1,01, bulk/tail ESS≥400. Исполнены все 9 code cells (21 rich output, без ошибок). Просмотрены все 8 PNG-графиков. Независимый review sol6.1 medium завершён, замечания исправлены. ZIP 65 MB проверен и распакован, verify из копии PASS.

Даты NPZ исправлены на U10 без изменения count-массивов (SHA проверены). Модели/выбор после test не менялись. Рендеринг обновлён для месячного покрытия и переносимых SVG-шрифтов; manifest и notebook актуальны. Исполнение notebook использует IPython in-process с memory history для sandbox; обычный Jupyter тоже поддерживается. Кэши можно перестроить. Нейросетевую очистку не возобновлять автоматически.
