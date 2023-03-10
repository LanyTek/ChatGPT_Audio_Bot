#!/bin/bash

python3 api_app.py &

# wait for api server to start
sleep 5

# load all the models
curl --max-time 300 "http://localhost:10000"

# run the chatbot web app
streamlit run chatbot.py