import streamlit as st
import pandas as pd
import calendar
import os

class ExpenseTracker:
    def __init__(self, csv_path='expenses.csv'):
        self.csv_path = csv_path
        self.load_data()

    def load_data(self):
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
            display_df['cost'] = display_df['cost'].apply(lambda x: f"${x:,.2f}")
            
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
            
            print(month_expenses.groupby('rate').groups)
            print(freq_totals)
            freq_data = pd.DataFrame({
                'Amount': freq_totals
            })
            freq_data.index = ['One-time', 'Daily', 'Weekly', 'Monthly']
            st.bar_chart(freq_data)
        else:
            st.info("No expenses recorded for this month")

if __name__ == "__main__":
    main()