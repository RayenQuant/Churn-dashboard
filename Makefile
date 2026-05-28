.PHONY: install run clean

install:
	pip install -r requirements.txt

run:
	streamlit run app.py --server.port 8501

clean:
	rm -f churn.db customer_churn.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
