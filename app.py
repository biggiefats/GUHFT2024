import streamlit as st
import pandas as pd
import calendar
import os
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ExpenseTracker:
    def __init__(self, csv_path='expenses.csv'):
        """
        Creates the big expense machine, for big expense people.
        """
        self.csv_path = csv_path
        self.load_data()

    def load_data(self):
        """
        Ensures data is loaded or can be loaded.
        """
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
        else:
            self.df = pd.DataFrame(columns=['name', 'cost', 'rate', 'priority', 'day', 'month'])
            self.save_data()

    def save_data(self):
        self.df.to_csv(self.csv_path, index=False)

    def add_expense(self, name, cost, rate, priority, day, month):
        new_expense = pd.DataFrame([{
            'name': name,
            'cost': cost,
            'rate': rate,
            'priority': priority,
            'day': day,
            'month': month
        }])
        self.df = pd.concat([self.df, new_expense], ignore_index=True)
        self.save_data()

    def remove_expense(self, name, day, month):
        self.df = self.df[~((self.df['name'] == name) & 
                           (self.df['day'] == day) & 
                           (self.df['month'] == month))]
        self.save_data()

    def get_month_expenses(self, month):
        return self.df[self.df['month'] == month]

    def get_day_expenses(self, day, month):
        return self.df[(self.df['day'] == day) & (self.df['month'] == month)]
    
    def most_possible_day(self):
        """
        Calculate the day and the percentage of spending on such a day.
        """
        max_freq = int()
        popularday = int()
        chance = float()

        day_counts = defaultdict(int)
        for i in range(len(self.df)):
            day = self.df['day'].iat[i]
            day_counts[str(day)] += 1

        for day in range(len(day_counts.keys())):
            freq = day_counts[str(day)] 
            max_freq = max(max_freq, freq)
            if max_freq == freq:
                popularday = day

        if len(self.df) > 0:  # Only perform analysis if there's data
            X = self.df[['cost','day','month']]
            y = self.df['day'].apply(lambda x: 1 if x == popularday else 0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, train_size=0.7)
            
            model = LogisticRegression(solver='liblinear', random_state=0)
            model.fit(X_train, y_train)
            y_predicted = model.predict(X_test)
            chance = round(accuracy_score(y_test, y_predicted) * 100)
        else:
            popularday = 1  # Default to Monday if no data
            chance = 0

        return popularday, chance
    
    def line_graph_analysis(self):
        """
        Creates line graph analysis with line of best fit
        """
        # Prepare data for plotting
        daily_expenses = self.df.groupby(['day'])['cost'].sum().reset_index()
        x = daily_expenses['day'].values.reshape(-1, 1)
        y = daily_expenses['cost'].values

        # Fit linear regression
        model = LinearRegression()
        model.fit(x, y)
        
        # Generate line of best fit
        y_pred = model.predict(x)
        
        return x.flatten(), y, y_pred

    def generate_heatmap_data(self, selected_month):
        """
        Generates heatmap data for spending patterns
        Returns a matrix of spending intensity for each day and time period
        """
        # Create time periods (4 quarters of the day)
        time_periods = ['Morning (12am-6am)', 'Day (6am-12pm)', 
                       'Afternoon (12pm-6pm)', 'Evening (6pm-12am)']
        
        # Initialize the heatmap matrix (28 days x 4 time periods)
        heatmap_data = np.zeros((28, 4))
        
        # Get month's expenses
        month_expenses = self.df[self.df['month'] == selected_month]
        
        # Calculate total spending for normalization
        max_spending = month_expenses['cost'].max() if not month_expenses.empty else 1
        
        # Simulate time distribution (since we don't have actual time data)
        # We'll distribute expenses across time periods based on priority
        # High priority -> Morning/Evening, Medium -> Day, Low -> Afternoon
        for _, expense in month_expenses.iterrows():
            day_idx = int(expense['day']) - 1
            cost = expense['cost']
            priority = expense['priority']
            
            # Distribute cost based on priority
            if priority == 'H':
                heatmap_data[day_idx, 0] += cost * 0.4  # Morning
                heatmap_data[day_idx, 3] += cost * 0.6  # Evening
            elif priority == 'M':
                heatmap_data[day_idx, 1] += cost * 0.7  # Day
                heatmap_data[day_idx, 2] += cost * 0.3  # Afternoon
            else:  # Low priority
                heatmap_data[day_idx, 1] += cost * 0.3  # Day
                heatmap_data[day_idx, 2] += cost * 0.7  # Afternoon
        
        # Normalize the data
        if max_spending > 0:
            heatmap_data = heatmap_data / max_spending
        
        return heatmap_data, time_periods

def create_month_calendar(tracker, month):
    days = list(range(1, 29))
    weeks = [days[i:i + 7] for i in range(0, 28, 7)]
    
    # Get month's expenses
    month_data = tracker.get_month_expenses(month)
    
    # Headers
    cols = st.columns(7)
    for i, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
        cols[i].markdown(f"**{day}**")
    
    # Calendar grid
    for week in weeks:
        cols = st.columns(7)
        for i, day in enumerate(week):
            with cols[i]:
                day_expenses = month_data[month_data['day'] == day]
                total = day_expenses['cost'].sum()
                
                # Get priority indicator
                if not day_expenses.empty:
                    if 'H' in day_expenses['priority'].values:
                        indicator = "🔴"
                    elif 'M' in day_expenses['priority'].values:
                        indicator = "🟡"
                    else:
                        indicator = "🟢"
                    display_text = f"{day}\n£{total:.0f}"
                else:
                    indicator = "⚪"
                    display_text = str(day)
                
                if st.button(f"{indicator} {display_text}", key=f"day_{month}_{day}"):
                    st.session_state.selected_day = day
                    st.session_state.selected_month = month

def main():
    st.set_page_config(page_title="Monthly Expense Tracker", layout="wide")
    st.title("💰 Monthly Expense Tracker")
    
    tracker = ExpenseTracker()
    popularday, chance = tracker.most_possible_day()
    
    # Month selection
    selected_month = st.selectbox(
        "Select Month",
        options=range(1, 13),
        format_func=lambda x: calendar.month_name[x],
        key="month_selector"
    )
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📅 {calendar.month_name[selected_month]}")
        create_month_calendar(tracker, selected_month)
        
        # Add line graph analysis
        st.subheader("Daily Expense Trend")
        x, y, y_pred = tracker.line_graph_analysis()
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Day': x,
            'Actual Expenses': y,
            'Trend Line': y_pred
        })
        
        # Plot both lines
        st.line_chart(plot_data.set_index('Day'))
        
        # Add heatmap
        st.subheader("Spending Intensity Heatmap")
        heatmap_data, time_periods = tracker.generate_heatmap_data(selected_month)
        
        # Create a DataFrame for the heatmap
        heatmap_df = pd.DataFrame(
            heatmap_data,
            columns=time_periods,
            index=[f"Day {i+1}" for i in range(28)]
        )
        
        # Custom CSS for the heatmap
        st.markdown("""
        <style>
        .heatmap {
            font-family: monospace;
            white-space: pre;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display heatmap using st.dataframe with custom styling
        st.dataframe(
            heatmap_df.style
            .background_gradient(cmap='YlOrRd')
            .format("{:.2%}")
            .set_properties(**{'width': '100px'})
            .set_caption("Spending Intensity (% of maximum daily spending)"),
            height=600
        )
    
    with col2:
        st.subheader("Add Expense")
        with st.form("expense_form"):
            name = st.text_input("Name")
            cost = st.number_input("Amount (£)", min_value=0.0, step=0.01)
            rate = st.selectbox("Frequency", 
                              options=['One-time', 'Daily', 'Weekly', 'Monthly'],
                              format_func=lambda x: x)
            priority = st.selectbox("Priority",
                                  options=['Low', 'Medium', 'High'],
                                  format_func=lambda x: x)
            day = st.number_input("Day", min_value=1, max_value=28,
                                value=getattr(st.session_state, 'selected_day', 1))
            
            if st.form_submit_button("Add Expense"):
                if name and cost > 0:
                    # Convert priority and rate to single letters
                    priority_map = {'Low': 'L', 'Medium': 'M', 'High': 'H'}
                    rate_map = {'One-time': 'O', 'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
                    
                    tracker.add_expense(
                        name, cost, 
                        rate_map[rate],
                        priority_map[priority],
                        day, selected_month
                    )
                    st.success("Expense added!")
                else:
                    st.error("Please fill all fields")
    
    # Show selected day's expenses
    if hasattr(st.session_state, 'selected_day'):
        day = st.session_state.selected_day
        month = st.session_state.selected_month
        
        st.subheader(f"Expenses for {calendar.month_name[month]}, Day {day}")
        
        day_expenses = tracker.get_day_expenses(day, month)
        if not day_expenses.empty:
            # Format for display
            display_df = day_expenses.copy()
            display_df['cost'] = display_df['cost'].apply(lambda x: f"£{x:,.2f}")
            
            # Convert codes to full text
            display_df['priority'] = display_df['priority'].map({
                'L': '🟢 Low', 'M': '🟡 Medium', 'H': '🔴 High'
            })
            display_df['rate'] = display_df['rate'].map({
                'O': 'One-time', 'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'
            })
            
            st.table(display_df[['name', 'cost', 'priority', 'rate']])
            
            # Delete functionality
            if st.button("Remove an expense"):
                expense = st.selectbox("Select expense to remove",
                                     day_expenses['name'].tolist())
                if expense and st.button("Confirm Delete"):
                    tracker.remove_expense(expense, day, month)
                    st.success("Expense removed!")
                    st.experimental_rerun()
        else:
            st.info("No expenses for this day")
    
    # Monthly summary in sidebar
    with st.sidebar:
        st.subheader("📊 Month Summary")
        month_expenses = tracker.get_month_expenses(selected_month)
        
        if not month_expenses.empty:
            total = month_expenses['cost'].sum()
            st.metric("Total Expenses", f"£{total:,.2f}")
            st.metric("Day Most Likely To Spend", f"{['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][popularday-1]}")
            st.metric("Chance of Spending", f"{chance}%")
            
            # Expenses by priority
            st.subheader("By Priority")
            priority_totals = month_expenses.groupby('priority')['cost'].sum()
            priority_data = pd.DataFrame({
                'Amount': priority_totals
            }).reindex(['L', 'M', 'H'])
            priority_data.index = ['Low', 'Medium', 'High']
            st.bar_chart(priority_data)
            
            # Expenses by frequency
            st.subheader("By Frequency")
            freq_totals = month_expenses.groupby('rate')['cost'].sum()
            
            # Create a DataFrame with all possible rate categories
            rate_categories = {'O': 'One-time', 'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}
            freq_data = pd.DataFrame(index=['O', 'D', 'W', 'M'])
            freq_data['Amount'] = freq_totals
            freq_data = freq_data.fillna(0)
            
            # Map the index to full names
            freq_data.index = [rate_categories[rate] for rate in freq_data.index]
            
            st.bar_chart(freq_data)
        else:
            st.info("No expenses recorded for this month")

if __name__ == "__main__":
    main()