import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("soapnutshistory.csv")
df

df.isnull().sum()

df.dropna(subset=['Organic Conversion Percentage'], inplace=True)

df

for i in df.columns:
  for k in df.loc[:,i]:
    if type(k)==float and df[i].isnull().sum()>0:
      df.fillna(np.mean(df[i]),inplace=True)


df.isnull().sum()

df.head(50)

df_col_subset=df.loc[:, ["Organic Conversion Percentage","Ad Conversion Percentage","Total Profit","Total Sales"]]

for i in df.index:
  row=True
  for k in df_col_subset.columns:
    if df.loc[i,k]!=0:
      row=False
  if row==True:
    df.drop(i,inplace=True)


df.head(50)

## Total sales vs pridicted sales

plt.scatter(df["Total Sales"],df["Predicted Sales"])
plt.xlabel("Total Sales")
plt.ylabel("Predicted Sales")
plt.show()

print(df.loc[:,"Total Sales"] < 0)
for i in df.index:
  if df.loc[i,"Total Sales"] < 0:
    print(i)

df.drop(59,inplace=True)

for i in df.index:
  if df.loc[i,"Total Sales"] > 200:
    df.drop(i,inplace=True)

df.loc[:,["Total Sales","Predicted Sales"]].corr()

plt.scatter(df["Total Sales"],df["Predicted Sales"])
plt.xlabel("Total Sales")
plt.ylabel("Predicted Sales")
plt.grid(True)
plt.show()

## **product price vs total sales**

plt.scatter(df["Product Price"],df["Total Sales"])
plt.xlabel("Product Price")
plt.ylabel("Total Sales")
plt.show()

df.loc[:,["Product Price","Total Sales"]].corr()

for i in df.index:
  if df.loc[i,'Product Price'] < 13:
    print(f"index: {i} ,value: {df.loc[i,'Product Price']}")

df.drop(65,inplace=True)

plt.scatter(df["Product Price"],df["Total Sales"])
plt.xlabel("Product Price")
plt.ylabel("Total Sales")
plt.show()

# Product Price vs Organic Conversion **Percentage**

plt.scatter(df["Product Price"],df["Organic Conversion Percentage"])
plt.xlabel("Product Price")
plt.ylabel("Organic Conversion Percentage")
plt.grid(True)
plt.show()

df.loc[:,["Product Price","Organic Conversion Percentage"]].corr()

for i in df.index:
  if df.loc[i,'Product Price'] < 14:
    print(f"index: {i} ,value: {df.loc[i,'Product Price']}")

for i in df.index:
  if df.loc[i,'Organic Conversion Percentage'] > 100:
    print(f"index: {i} ,value: {df.loc[i,'Organic Conversion Percentage']}")

df.drop(100,inplace=True)
df.drop(139,inplace=True)
df.drop(153,inplace=True)

plt.scatter(df["Product Price"],df["Organic Conversion Percentage"])
plt.xlabel("Product Price")
plt.ylabel("Organic Conversion Percentage")
plt.grid(True)
plt.show()

## Product Price vs Ad Conversion **Percentage**

plt.scatter(df["Product Price"],df["Ad Conversion Percentage"])
plt.xlabel("Product Price")
plt.ylabel("Ad Conversion Percentage")
plt.show()

df.loc[:,["Product Price","Ad Conversion Percentage"]].corr()

for i in df.index:
  if df.loc[i,"Product Price"] < 13:
    df.drop(i,inplace=True)
  if df.loc[i,"Ad Conversion Percentage"] > 30:
    df.drop(i,inplace=True)

plt.scatter(df["Product Price"],df["Ad Conversion Percentage"])
plt.xlabel("Product Price")
plt.ylabel("Ad Conversion Percentage")
plt.show()

## **product price vs predicted sales**

plt.scatter(df["Product Price"],df["Predicted Sales"])
plt.xlabel("Product Price")
plt.ylabel("Predicted Sales")
plt.show()

df.loc[:,["Product Price","Predicted Sales"]].corr()

for i in df.index:
  if df.loc[i,"Product Price"] < 13:
    df.drop(i,inplace=True)

plt.scatter(df["Product Price"],df["Predicted Sales"])
plt.xlabel("Product Price")
plt.ylabel("Predicted Sales")
plt.show()

## final overview of **data**

df

df.describe()

plt.scatter(df["Product Price"],df["Total Sales"])
plt.xlabel("Product Price")
plt.ylabel("Total Sales")
plt.show()

plt.scatter(df["Predicted Sales"],df["Total Sales"])
plt.xlabel("Predicted Sales")
plt.ylabel("Total Sales")
plt.show()

plt.scatter(df["Product Price"],df["Organic Conversion Percentage"])
plt.xlabel("Product Price")
plt.ylabel("Organic Conversion Percentage")
plt.show()

plt.scatter(df["Product Price"],df["Ad Conversion Percentage"])
plt.xlabel("Product Price")
plt.ylabel("Ad Conversion Percentage")
plt.show()

plt.scatter(df["Product Price"],df["Predicted Sales"])
plt.xlabel("Product Price")
plt.ylabel("Predicted Sales")
plt.show()

# if not working install these 
# !pip install gym
# !pip install stable-baselines3[extra]
# !pip install shimmy>=2.0
import gym
import pandas as pd
from gym import spaces
import numpy as np
# import matplotlib.pyplot as plt
from stable_baselines3 import PPO

file_path = '/content/woolballhistory.csv'
df = pd.read_csv(file_path)

def clean_data(df):
    df_cleaned = df.dropna(subset=['Product Price', 'Total Sales']).copy()
    df_cleaned = df_cleaned[df_cleaned['Total Sales'] > 0]

    for col in ['Organic Conversion Percentage', 'Ad Conversion Percentage']:
        if col in df_cleaned.columns:
            lower_bound = df_cleaned[col].quantile(0.01)
            upper_bound = df_cleaned[col].quantile(0.99)
            df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)


    if 'Report Date' in df_cleaned.columns:
        df_cleaned['Report Date'] = pd.to_datetime(df_cleaned['Report Date'])
        df_cleaned = df_cleaned.sort_values(by='Report Date')

    return df_cleaned


df = clean_data(df)

class market(gym.Env):
  def __init__(self,df):
    super(market,self).__init__()
    self.df = df
    self.minprice=df['Product Price'].min()
    self.maxprice=df['Product Price'].max()
    self.medianprice=df['Product Price'].median()

    self.action_space=spaces.Discrete(11)
    self.observation_space=spaces.Box(
        low=np.array([self.minprice,0]),
        high=np.array([self.maxprice,df['Total Sales'].max()]),
        dtype=np.float32
    )

    self.current_price=None
    self.current_sales=None

  def reset(self):
    random_index = np.random.randint(0, len(self.df))
    self.current_price=self.df.iloc[random_index]['Product Price']
    self.current_sales=self.df.iloc[random_index]['Total Sales']
    return np.array([self.current_price,self.current_sales])

  def step(self,action):
    price_incre=(action-2)
    self.current_price= np.clip(self.current_price + price_incre,self.minprice,self.maxprice)

    new_sales=self.get_sales(self.current_price)
    reward=self.calculate_reward(new_sales)
    self.current_sales=new_sales

    done=self.current_price == self.minprice or self.current_price == self.maxprice


    return np.array([self.current_price,self.current_sales]),reward,done,{}

  def get_sales(self,price):
    prices = self.df['Product Price'].values
    sales = self.df['Total Sales'].values

    print(f"Price: {price}")
    print(f"Price Range in Data: [{prices.min()}, {prices.max()}]")
    print(f"Sales Range in Data: [{sales.min()}, {sales.max()}]")

    interpolated_sales = np.interp(price, prices, sales)
    print(f"Interpolated Sales: {interpolated_sales}")

    return max(0, interpolated_sales)

  def calculate_reward(self, sales):
    reward = sales
    if self.current_price > self.medianprice:
      reward += (self.current_price - self.medianprice)*0.1
    if sales <= 0:
        reward -= 50
    if sales < self.current_sales:
      reward -= (self.current_sales - sales)*0.5
    return reward




env = market(df)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
obs = env.reset()
for _ in range(20):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Price: {obs[0]:.2f}, Sales: {obs[1]:.0f}, Reward: {reward:.2f}")

import matplotlib.pyplot as plt

def visualize_episode(env, policy=None, max_steps=10000):

    prices = []
    sales = []
    rewards = []

    # Reset the environment
    obs = env.reset()

    for step in range(max_steps):
        # Use a random policy if no policy is specified
        action = env.action_space.sample() if policy is None else policy(obs)

        # Step the environment
        obs, reward, done, _ = env.step(action)

        # Record data
        prices.append(obs[0])  # Current price
        sales.append(obs[1])  # Current sales
        rewards.append(reward)  # Reward

        # If the episode ends, stop simulation
        if done:
            break

    # Check if there’s enough data to plot
    if len(prices) <= 1:
        print("Not enough data to visualize. Please check your environment or policy logic.")
        return

    # Plot the results
    plt.figure(figsize=(15, 5))

    # Price plot
    plt.subplot(1, 3, 1)
    plt.plot(prices, marker='o', label='Price', color='blue')
    plt.axhline(env.medianprice, color='red', linestyle='--', label='Median Price')
    plt.title("Price Over Steps")
    plt.xlabel("Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()

    # Sales plot
    plt.subplot(1, 3, 2)
    plt.plot(sales, marker='o', label='Sales', color='green')
    plt.title("Sales Over Steps")
    plt.xlabel("Steps")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid()

    # Reward plot
    plt.subplot(1, 3, 3)
    plt.plot(rewards, marker='o', label='Reward', color='orange')
    plt.title("Reward Over Steps")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Initialize the environment and test the visualization
env = market(df)
visualize_episode(env, max_steps=10000)



