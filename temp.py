import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns


# Load Excel file into DataFrame
df = pd.read_excel('Entertainer Data.xlsx')

# Check column names
print(df.columns)

# Strip any leading or trailing whitespaces from column names
df.columns = df.columns.str.strip()

# Descriptive Statistics for Birth Year
birth_year_stats = df['Birth Year'].describe()
mean_birth_year = df['Birth Year'].mean()
median_birth_year = df['Birth Year'].median()
std_birth_year = df['Birth Year'].std()

print("Descriptive Statistics for Birth Year:")
print(birth_year_stats)
print(f"Mean Birth Year: {mean_birth_year}")
print(f"Median Birth Year: {median_birth_year}")
print(f"Standard Deviation of Birth Year: {std_birth_year}")

# Explore Gender Distribution
gender_counts = df['Gender'].value_counts()
print("\nGender Distribution among Entertainers:")
print(gender_counts)

if 'Year of Breakthrough/#1 Hit/Award Nomination' in df.columns:
    # Group by Breakthrough Year and count occurrences
    breakthrough_counts = df['Year of Breakthrough/#1 Hit/Award Nomination'].value_counts().sort_index()

    # Plotting trends
    plt.figure(figsize=(10, 6))
    breakthrough_counts.plot(kind='bar', color='skyblue')
    plt.title('Trends in Breakthrough Years')
    plt.xlabel('Breakthrough Year')
    plt.ylabel('Number of Entertainers')
    plt.grid(axis='y')
    plt.show()

else:
    print("Breakthrough Year information not available in the dataset.")

current_year = pd.Timestamp.now().year
# Calculate the age at which entertainers achieve their first major award
df['Age at First Award'] = df['Year of First Oscar/Grammy/Emmy'] - df['Birth Year']
average_age_first_award = df['Age at First Award'].mean()
print(f'Average age at which entertainers achieve their first major award: {average_age_first_award:.2f}')

# Investigate the correlation between breakthrough year and award nominations
# Assuming 'Year of Breakthrough/#1 Hit/Award Nomination' represents the breakthrough year
# and that 'Award Nominations' column exists in the dataset
# If 'Award Nominations' does not exist, replace with the correct column name or add the relevant column
correlation = df['Year of Breakthrough/#1 Hit/Award Nomination'].corr(df['Year of First Oscar/Grammy/Emmy'])
print(f'Correlation between breakthrough year and award nominations: {correlation:.2f}')

current_year = pd.Timestamp.now().year
# Prepare the data for survival analysis
# If Year of Death is NaN, it means the entertainer is still alive
df['Year of Death'].fillna(current_year, inplace=True)

# Calculate the duration (years active)
df['Years Active'] = df['Year of Last Major Work (arguable)'] - df['Year of Breakthrough/#1 Hit/Award Nomination']
df['Event'] = df['Year of Death'] != current_year  # Event occurred if the entertainer is not alive

# Kaplan-Meier Fitter
kmf = KaplanMeierFitter()

# Fit the data
kmf.fit(df['Years Active'], event_observed=df['Event'])

# Plot the survival function
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Survival Probability of Entertainers')
plt.xlabel('Years Active')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

# Ensure that the columns are correctly interpreted
print(df.columns)

# Check for missing values in the Gender column
print(df['Gender'].isnull().sum())

# Drop rows with missing Gender values (if any)
df = df.dropna(subset=['Gender'])

# Gender Distribution Visualization
plt.figure(figsize=(14, 6))

# Bar Chart for Gender Distribution
plt.subplot(1, 2, 1)
sns.countplot(x=df['Gender'], palette='pastel')
plt.title('Gender Distribution - Bar Chart')
plt.xlabel('Gender')
plt.ylabel

# Pie Chart for Gender Distribution
plt.subplot(1, 2, 2)
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Gender Distribution - Pie Chart')

plt.tight_layout()
plt.show()

# Extract relevant columns for time series analysis
breakthrough_years = df['Year of Breakthrough/#1 Hit/Award Nomination']
last_major_work_years = df['Year of Last Major Work (arguable)']

# Plot the breakthrough years over time
plt.figure(figsize=(14, 7))
sns.histplot(breakthrough_years, bins=range(breakthrough_years.min(), breakthrough_years.max() + 1), kde=False)
plt.title('Number of Breakthroughs Over the Years')
plt.xlabel('Year of Breakthrough')
plt.ylabel('Number of Entertainers')
plt.show()

# Plot the last major work years over time
plt.figure(figsize=(14, 7))
sns.histplot(last_major_work_years, bins=range(last_major_work_years.min(), last_major_work_years.max() + 1), kde=False)
plt.title('Number of Last Major Works Over the Years')
plt.xlabel('Year of Last Major Work')
plt.ylabel('Number of Entertainers')
plt.show()

# Calculate moving averages
breakthrough_years_ma = breakthrough_years.rolling(window=5).mean()
last_major_work_years_ma = last_major_work_years.rolling(window=5).mean()

# Plot the moving averages
plt.figure(figsize=(14, 7))
plt.plot(breakthrough_years, label='Breakthrough Years', alpha=0.5)
plt.plot(breakthrough_years_ma, label='Breakthrough Years (5-Year MA)', linewidth=2)
plt.title('Breakthrough Years with 5-Year Moving Average')
plt.xlabel('Index')
plt.ylabel('Year')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(last_major_work_years, label='Last Major Work Years', alpha=0.5)
plt.plot(last_major_work_years_ma, label='Last Major Work Years (5-Year MA)', linewidth=2)
plt.title('Last Major Work Years with 5-Year Moving Average')
plt.xlabel('Index')
plt.ylabel('Year')
plt.legend()
plt.show()

# Exponential Smoothing
breakthrough_years_ewm = breakthrough_years.ewm(span=5, adjust=False).mean()
last_major_work_years_ewm = last_major_work_years.ewm(span=5, adjust=False).mean()

# Plot the exponential smoothing
plt.figure(figsize=(14, 7))
plt.plot(breakthrough_years, label='Breakthrough Years', alpha=0.5)
plt.plot(breakthrough_years_ewm, label='Breakthrough Years (5-Year EWM)', linewidth=2)
plt.title('Breakthrough Years with 5-Year Exponential Smoothing')
plt.xlabel('Index')
plt.ylabel('Year')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(last_major_work_years, label='Last Major Work Years', alpha=0.5)
plt.plot(last_major_work_years_ewm, label='Last Major Work Years (5-Year EWM)', linewidth=2)
plt.title('Last Major Work Years with 5-Year Exponential Smoothing')
plt.xlabel('Index')
plt.ylabel('Year')
plt.legend()
plt.show()

df['Age at Breakthrough'] = df['Year of Breakthrough/#1 Hit/Award Nomination'] - df['Birth Year']

# Calculate the average age at breakthrough
average_age_at_breakthrough = df['Age at Breakthrough'].mean()
print(f'Average age at breakthrough: {average_age_at_breakthrough:.2f} years')

# Compare average age at breakthrough across different genders
average_age_by_gender = df.groupby('Gender')['Age at Breakthrough'].mean()
print('Average age at breakthrough by gender:')
print(average_age_by_gender)

# Compare average age at breakthrough across different entertainment types (Breakthrough Name)
average_age_by_entertainment_type = df.groupby('Breakthrough Name')['Age at Breakthrough'].mean()
print('Average age at breakthrough by entertainment type:')
print(average_age_by_entertainment_type)

# Visualization
plt.figure(figsize=(14, 7))

# Bar chart for average age at breakthrough by gender
plt.subplot(1, 2, 1)
sns.barplot(x=average_age_by_gender.index, y=average_age_by_gender.values, palette='pastel')
plt.title('Average Age at Breakthrough by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Age')

# Bar chart for average age at breakthrough by entertainment type
plt.subplot(1, 2, 2)
sns.barplot(x=average_age_by_entertainment_type.index, y=average_age_by_entertainment_type.values, palette='pastel')
plt.title('Average Age at Breakthrough by Entertainment Type')
plt.xlabel('Entertainment Type')
plt.ylabel('Average Age')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

df = df.dropna(subset=['Year of Breakthrough/#1 Hit/Award Nomination', 'Year of First Oscar/Grammy/Emmy', 'Year of Last Major Work (arguable)'])

# Rename columns for easier reference
df.rename(columns={
    'Year of Breakthrough/#1 Hit/Award Nomination': 'Breakthrough Year',
    'Year of First Oscar/Grammy/Emmy': 'First Award Year',
    'Year of Last Major Work (arguable)': 'Last Major Work Year'
}, inplace=True)

# Correlation between breakthrough year and first award year
correlation_breakthrough_award = df['Breakthrough Year'].corr(df['First Award Year'])
print(f'Correlation between breakthrough year and first award year: {correlation_breakthrough_award:.2f}')

# Correlation between breakthrough year and last major work year
correlation_breakthrough_last_work = df['Breakthrough Year'].corr(df['Last Major Work Year'])
print(f'Correlation between breakthrough year and last major work year: {correlation_breakthrough_last_work:.2f}')

# Scatter plot to visualize the relationship between breakthrough year and first award year
plt.figure(figsize=(14, 7))
sns.scatterplot(x=df['Breakthrough Year'], y=df['First Award Year'], hue=df['Gender'], palette='pastel')
plt.title('Breakthrough Year vs. First Award Year')
plt.xlabel('Breakthrough Year')
plt.ylabel('First Award Year')
plt.show()

# Scatter plot to visualize the relationship between breakthrough year and last major work year
plt.figure(figsize=(14, 7))
sns.scatterplot(x=df['Breakthrough Year'], y=df['Last Major Work Year'], hue=df['Gender'], palette='pastel')
plt.title('Breakthrough Year vs. Last Major Work Year')
plt.xlabel('Breakthrough Year')
plt.ylabel('Last Major Work Year')
plt.show()


# Prepare the data
df['Year of Death'].fillna(pd.Timestamp.now().year, inplace=True)
df['Survival Time'] = df['Year of Death'] - df['Birth Year']
df['Event'] = df['Year of Death'] != pd.Timestamp.now().year

# Fit the Kaplan-Meier estimator
kmf = KaplanMeierFitter()
kmf.fit(durations=df['Survival Time'], event_observed=df['Event'])

# Extract the survival function
survival_df = kmf.survival_function_.reset_index()
survival_df.columns = ['Survival Time', 'Survival Probability']

df.to_csv('Processed_Entertainer_Data.csv', index=False)
