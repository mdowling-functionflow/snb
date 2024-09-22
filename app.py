import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

class RLAgent:
    def __init__(self, n_assets, strategy='basic', learning_rate=0.01, exploration_rate=0.1, memory_length=10):
        self.n_assets = n_assets
        self.weights = np.ones(n_assets) / n_assets
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.memory_length = memory_length

    def get_action(self, market_state=None):
        if np.random.random() < self.exploration_rate:
            action = np.random.dirichlet(np.ones(self.n_assets))
        else:
            if self.strategy == 'trend_following' and market_state is not None:
                action = self.trend_following_action(market_state)
            elif self.strategy == 'mean_reversion' and market_state is not None:
                action = self.mean_reversion_action(market_state)
            else:
                action = self.weights
        return action

    def trend_following_action(self, market_state):
        trend = market_state.pct_change(periods=self.memory_length).iloc[-1]
        action = np.clip(trend, 0, 1)
        return action / action.sum()

    def mean_reversion_action(self, market_state):
        mean = market_state.rolling(window=self.memory_length).mean().iloc[-1]
        deviation = (market_state.iloc[-1] - mean) / mean
        action = 1 - np.clip(deviation, -1, 1)
        return action / action.sum()

    def update(self, reward, action):
        self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * action

def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def simulate_strategies(data, rl_agent):
    returns = data.pct_change().fillna(0)
    
    # Buy and Hold strategy
    buy_and_hold = (1 + returns).cumprod()
    
    # RL strategy
    rl_portfolio = pd.Series(index=data.index, dtype=float)
    rl_portfolio.iloc[0] = 1.0  # Start with $1
    rl_weights = []

    for i in range(1, len(data)):
        action = rl_agent.get_action(data.iloc[:i])
        rl_weights.append(action)
        daily_return = (action * returns.iloc[i]).sum()
        rl_portfolio.iloc[i] = rl_portfolio.iloc[i-1] * (1 + daily_return)
        rl_agent.update(daily_return, action)
    
    return buy_and_hold, rl_portfolio, rl_weights

def main():
    st.title("Interactive Reinforcement Learning for Finance")

    st.write("""
    Explore how different reinforcement learning (RL) strategies perform in the stock market. 
    This app compares various RL approaches against a simple buy-and-hold strategy using real market data.
    """)

    default_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    selected_tickers = st.multiselect(
        "Select stocks for your portfolio (2-5 recommended)",
        options=default_tickers + ["NVDA", "TSLA", "JPM", "JNJ", "V", "PG", "DIS", "NFLX", "ADBE", "CRM"],
        default=default_tickers[:3]
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", datetime.now() - timedelta(days=365*2))
    with col2:
        end_date = st.date_input("End date", datetime.now())

    rl_strategy = st.selectbox(
        "Choose an RL strategy",
        options=["basic", "trend_following", "mean_reversion"],
        format_func=lambda x: x.replace('_', ' ').title()
    )

    st.subheader("Fine-tune your RL agent")
    learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)
    exploration_rate = st.slider("Exploration Rate", 0.0, 0.5, 0.1)
    memory_length = st.slider("Memory Length", 1, 30, 10)

    if st.button("Run Simulation"):
        if len(selected_tickers) < 2:
            st.error("Please select at least 2 stocks for your portfolio.")
        else:
            data = get_stock_data(selected_tickers, start_date, end_date)
            rl_agent = RLAgent(len(selected_tickers), rl_strategy, learning_rate, exploration_rate, memory_length)
            buy_and_hold, rl_performance, rl_weights = simulate_strategies(data, rl_agent)

            st.subheader("Strategy Performance")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(rl_performance, label=f"{rl_strategy.replace('_', ' ').title()} RL Strategy")
            ax.plot(buy_and_hold.mean(axis=1), label="Buy and Hold", linestyle='--')
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value ($)")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.subheader("Final Returns")
            st.write(f"{rl_strategy.replace('_', ' ').title()} RL Strategy: {(rl_performance.iloc[-1] / rl_performance.iloc[0] - 1):.2%}")
            st.write(f"Buy and Hold: {(buy_and_hold.iloc[-1].mean() / buy_and_hold.iloc[0].mean() - 1):.2%}")

            st.subheader("RL Agent's Portfolio Evolution")
            rl_weights_df = pd.DataFrame(rl_weights, index=data.index[1:], columns=selected_tickers)
            st.area_chart(rl_weights_df)

if __name__ == "__main__":
    main()
