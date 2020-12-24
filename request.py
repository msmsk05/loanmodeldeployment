import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'Dependents':1, 'ApplicantIncome':3000, 'CoapplicantIncome':2000, "LoanAmount":5000, "Loan_Amount_Term":360, "Credit_History":1})

print(r.json())