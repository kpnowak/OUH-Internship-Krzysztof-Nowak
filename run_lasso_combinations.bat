@echo off
echo Running all LASSO combinations for Breast classification dataset...
echo.

REM LASSO + LogisticRegression combinations
echo === LASSO + LogisticRegression ===
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model LogisticRegression --fusion average --n-features 8 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model LogisticRegression --fusion average --n-features 16 --single  
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model LogisticRegression --fusion average --n-features 32 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model LogisticRegression --fusion attention_weighted --n-features 8 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model LogisticRegression --fusion attention_weighted --n-features 16 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model LogisticRegression --fusion attention_weighted --n-features 32 --single

REM LASSO + RandomForestClassifier combinations  
echo.
echo === LASSO + RandomForestClassifier ===
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model RandomForestClassifier --fusion average --n-features 8 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model RandomForestClassifier --fusion average --n-features 16 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model RandomForestClassifier --fusion average --n-features 32 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model RandomForestClassifier --fusion attention_weighted --n-features 8 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model RandomForestClassifier --fusion attention_weighted --n-features 16 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model RandomForestClassifier --fusion attention_weighted --n-features 32 --single

REM LASSO + SVC combinations
echo.
echo === LASSO + SVC ===
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model SVC --fusion average --n-features 8 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model SVC --fusion average --n-features 16 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model SVC --fusion average --n-features 32 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model SVC --fusion attention_weighted --n-features 8 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model SVC --fusion attention_weighted --n-features 16 --single
python tuner_halving.py --dataset Breast --task clf --approach selectors --selector LASSO --model SVC --fusion attention_weighted --n-features 32 --single

echo.
echo All LASSO combinations completed!
echo Check the hp_best/ folder for results and tuner_logs/ for detailed logs.
pause 