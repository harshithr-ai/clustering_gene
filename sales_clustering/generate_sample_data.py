"""
Sample Data Generator
Creates synthetic datasets for testing the clustering platform
"""

import pandas as pd
import numpy as np

def generate_customer_segmentation_data(n_samples=1000):
    """
    Generate synthetic customer data for retail segmentation
    
    Creates 3 natural customer segments:
    1. High-value frequent buyers
    2. Occasional bargain hunters
    3. New/infrequent customers
    """
    np.random.seed(42)
    
    # Segment 1: High-value frequent buyers (30%)
    n_seg1 = int(n_samples * 0.3)
    seg1 = pd.DataFrame({
        'CustomerID': range(1, n_seg1 + 1),
        'Age': np.random.normal(45, 8, n_seg1),
        'AnnualIncome': np.random.normal(85000, 15000, n_seg1),
        'PurchaseFrequency': np.random.normal(24, 4, n_seg1),
        'AverageOrderValue': np.random.normal(250, 50, n_seg1),
        'RecencyDays': np.random.normal(15, 5, n_seg1),
        'CustomerLifetimeMonths': np.random.normal(48, 12, n_seg1),
        'WebsiteVisitsPerMonth': np.random.normal(20, 5, n_seg1),
        'EmailEngagementRate': np.random.normal(0.65, 0.10, n_seg1),
        'CustomerSatisfactionScore': np.random.normal(4.5, 0.3, n_seg1),
        'ReturnRate': np.random.normal(0.05, 0.02, n_seg1),
        'Segment': 'High-Value'
    })
    
    # Segment 2: Occasional bargain hunters (40%)
    n_seg2 = int(n_samples * 0.4)
    seg2 = pd.DataFrame({
        'CustomerID': range(n_seg1 + 1, n_seg1 + n_seg2 + 1),
        'Age': np.random.normal(35, 10, n_seg2),
        'AnnualIncome': np.random.normal(55000, 12000, n_seg2),
        'PurchaseFrequency': np.random.normal(8, 3, n_seg2),
        'AverageOrderValue': np.random.normal(80, 25, n_seg2),
        'RecencyDays': np.random.normal(45, 15, n_seg2),
        'CustomerLifetimeMonths': np.random.normal(24, 10, n_seg2),
        'WebsiteVisitsPerMonth': np.random.normal(8, 3, n_seg2),
        'EmailEngagementRate': np.random.normal(0.35, 0.12, n_seg2),
        'CustomerSatisfactionScore': np.random.normal(3.8, 0.4, n_seg2),
        'ReturnRate': np.random.normal(0.12, 0.04, n_seg2),
        'Segment': 'Bargain-Hunter'
    })
    
    # Segment 3: New/infrequent customers (30%)
    n_seg3 = n_samples - n_seg1 - n_seg2
    seg3 = pd.DataFrame({
        'CustomerID': range(n_seg1 + n_seg2 + 1, n_samples + 1),
        'Age': np.random.normal(28, 8, n_seg3),
        'AnnualIncome': np.random.normal(45000, 10000, n_seg3),
        'PurchaseFrequency': np.random.normal(3, 2, n_seg3),
        'AverageOrderValue': np.random.normal(120, 40, n_seg3),
        'RecencyDays': np.random.normal(120, 40, n_seg3),
        'CustomerLifetimeMonths': np.random.normal(8, 4, n_seg3),
        'WebsiteVisitsPerMonth': np.random.normal(4, 2, n_seg3),
        'EmailEngagementRate': np.random.normal(0.25, 0.10, n_seg3),
        'CustomerSatisfactionScore': np.random.normal(3.5, 0.5, n_seg3),
        'ReturnRate': np.random.normal(0.08, 0.03, n_seg3),
        'Segment': 'New-Customer'
    })
    
    # Combine and shuffle
    df = pd.concat([seg1, seg2, seg3], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add some categorical features
    df['PreferredChannel'] = np.random.choice(['Online', 'In-Store', 'Mobile'], n_samples, 
                                              p=[0.5, 0.3, 0.2])
    df['MembershipTier'] = np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], n_samples,
                                           p=[0.4, 0.3, 0.2, 0.1])
    
    # Add some missing values (5%)
    for col in ['Age', 'AnnualIncome', 'EmailEngagementRate']:
        missing_idx = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    # Clip values to realistic ranges
    df['Age'] = df['Age'].clip(18, 80)
    df['AnnualIncome'] = df['AnnualIncome'].clip(20000, 200000)
    df['PurchaseFrequency'] = df['PurchaseFrequency'].clip(0, None)
    df['AverageOrderValue'] = df['AverageOrderValue'].clip(10, None)
    df['RecencyDays'] = df['RecencyDays'].clip(1, 365)
    df['EmailEngagementRate'] = df['EmailEngagementRate'].clip(0, 1)
    df['CustomerSatisfactionScore'] = df['CustomerSatisfactionScore'].clip(1, 5)
    df['ReturnRate'] = df['ReturnRate'].clip(0, 0.5)
    
    return df

if __name__ == "__main__":
    # Generate sample dataset
    df = generate_customer_segmentation_data(1000)
    
    # Save with and without true labels
    df_with_labels = df.copy()
    df_without_labels = df.drop('Segment', axis=1)
    
    df_with_labels.to_csv('sample_data_with_labels.csv', index=False)
    df_without_labels.to_csv('sample_data.csv', index=False)
    
    print("Sample datasets generated:")
    print(f"- sample_data.csv ({len(df)} rows, for clustering)")
    print(f"- sample_data_with_labels.csv ({len(df)} rows, for validation)")
    print(f"\nDataset characteristics:")
    print(f"- 3 natural customer segments")
    print(f"- 11 numerical features + 2 categorical features")
    print(f"- ~5% missing values")
    print(f"\nTrue segments distribution:")
    print(df['Segment'].value_counts())
