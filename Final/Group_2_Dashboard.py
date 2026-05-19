import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & Rename Data ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'Dataset', 'olist_granular_dataset.csv')

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Missing core dataset: {DATA_PATH}")

df_raw = pd.read_csv(DATA_PATH)

# Clean date column parsing
date_cols = ['order_purchase_timestamp', 'order_approved_at',
             'order_delivered_carrier_date', 'order_delivered_customer_date',
             'order_estimated_delivery_date']
for col in date_cols:
    df_raw[col] = pd.to_datetime(df_raw[col], errors='coerce')

# Map raw headers to our clean, Title Case analytical schema
rename_dict = {
    'product_category_name_english': 'Product Category',
    'product_weight_g': 'Product Weight (g)',
    'product_length_cm': 'Product Length(cm)',
    'product_height_cm': 'Product Height(cm)',
    'product_width_cm': 'Product Width (cm)',
    'price': 'Price of Item',
    'freight_value': 'Freight Cost',
    'payment_type': 'Payment Type',
    'payment_value': 'Payment Value',
    'avg_review_score': 'review_score',
    'order_id': 'Order ID',
    'order_status': 'Order Status',
    'order_purchase_timestamp': 'Order Date',
    'order_approved_at': 'Approved Date of order',
    'order_delivered_carrier_date': 'Carrier Pickup Date',
    'order_delivered_customer_date': 'Delivery Date',
    'order_estimated_delivery_date': 'Estimated Delivery Date',
    'order_item_id': 'Number of Item/s',
    'customer_state': 'customer_state',
    'customer_city': 'customer_city'
}

df = df_raw.rename(columns=rename_dict)

# Impute critical columns for stable, crash-proof dashboard math
df['Product Category'] = df['Product Category'].fillna('Unknown')
df['Price of Item'] = df['Price of Item'].fillna(df['Price of Item'].median())
df['Freight Cost'] = df['Freight Cost'].fillna(df['Freight Cost'].median())
df['review_score'] = df['review_score'].fillna(4.0)

# Pre-calculate operational variables for visualization
df['YearMonth'] = df['Order Date'].dt.to_period('M')
df['Delivery Delay Days'] = (df['Delivery Date'] - df['Estimated Delivery Date']).dt.days
df['Transit Time Days'] = (df['Delivery Date'] - df['Order Date']).dt.days

# Populate filter lists
available_categories = sorted(df['Product Category'].dropna().unique())
available_states = sorted(df['customer_state'].dropna().unique())

print(f"SUCCESS: Aligned and loaded dashboard database. Total items: {len(df):,}")

# ── 2. Build Premium Dark-Mode Dash App ──
app = Dash(__name__)
app.title = "Olist Executive E-Commerce Command Center"

# Custom dark-theme styling constants
bg_main = '#0f172a'  # Slate 900
bg_card = '#1e293b'  # Slate 800
border_card = '#334155'  # Slate 700
text_main = '#f8fafc'  # Slate 50
text_sec = '#94a3b8'  # Slate 400
accent_color = '#38bdf8'  # Sky 400

app.layout = html.Div([

    # Executive Header
    html.Div([
        html.H1("OLIST MARKETPLACE EXECUTIVE PERFORMANCE DASHBOARD",
                style={'textAlign': 'center', 'color': text_main, 'fontSize': '28px', 
                       'letterSpacing': '1.5px', 'fontWeight': 'bold', 'margin': '0 0 20px 0'})
    ], style={'padding': '20px 10px', 'backgroundColor': bg_main}),

    # Interactive Glassmorphic Filters Block
    html.Div([
        html.Div([
            html.Label("🔍 Filter by Product Category", style={'color': text_main, 'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': c.replace('_', ' ').title(), 'value': c} for c in available_categories],
                value=[],
                multi=True,
                placeholder="All Categories Active",
                style={'backgroundColor': bg_main, 'color': '#000000'}
            )
        ], style={'flex': '1', 'minWidth': '300px', 'margin': '10px'}),

        html.Div([
            html.Label("📍 Filter by Customer State", style={'color': text_main, 'fontWeight': 'bold', 'marginBottom': '8px', 'display': 'block'}),
            dcc.Dropdown(
                id='state-filter',
                options=[{'label': s, 'value': s} for s in available_states],
                value=[],
                multi=True,
                placeholder="All States Active",
                style={'backgroundColor': bg_main, 'color': '#000000'}
            )
        ], style={'flex': '1', 'minWidth': '300px', 'margin': '10px'})
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'backgroundColor': bg_card, 
              'borderRadius': '12px', 'border': f'1px solid {border_card}', 
              'margin': '0 20px 20px 20px', 'padding': '15px'}),

    # Real-Time KPI Metric Cards Row
    html.Div([
        html.Div([
            html.H3("TOTAL REVENUE", style={'fontSize': '12px', 'color': text_sec, 'margin': '0 0 5px 0', 'letterSpacing': '1px'}),
            html.Div(id='kpi-revenue', style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#10b981'})  # Emerald
        ], style={'flex': '1', 'minWidth': '200px', 'backgroundColor': bg_card, 'border': f'1px solid {border_card}', 'borderRadius': '10px', 'padding': '15px', 'margin': '10px', 'textAlign': 'center'}),

        html.Div([
            html.H3("TOTAL TRANSACTIONS", style={'fontSize': '12px', 'color': text_sec, 'margin': '0 0 5px 0', 'letterSpacing': '1px'}),
            html.Div(id='kpi-orders', style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#6366f1'})  # Indigo
        ], style={'flex': '1', 'minWidth': '200px', 'backgroundColor': bg_card, 'border': f'1px solid {border_card}', 'borderRadius': '10px', 'padding': '15px', 'margin': '10px', 'textAlign': 'center'}),

        html.Div([
            html.H3("CUSTOMER REVIEW AVG", style={'fontSize': '12px', 'color': text_sec, 'margin': '0 0 5px 0', 'letterSpacing': '1px'}),
            html.Div(id='kpi-satisfaction', style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#f59e0b'})  # Amber
        ], style={'flex': '1', 'minWidth': '200px', 'backgroundColor': bg_card, 'border': f'1px solid {border_card}', 'borderRadius': '10px', 'padding': '15px', 'margin': '10px', 'textAlign': 'center'}),

        html.Div([
            html.H3("AVG SHIPPED TRANSIT TIME", style={'fontSize': '12px', 'color': text_sec, 'margin': '0 0 5px 0', 'letterSpacing': '1px'}),
            html.Div(id='kpi-transit', style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#ec4899'})  # Pink
        ], style={'flex': '1', 'minWidth': '200px', 'backgroundColor': bg_card, 'border': f'1px solid {border_card}', 'borderRadius': '10px', 'padding': '15px', 'margin': '10px', 'textAlign': 'center'})
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '0 10px 10px 10px'}),

    # Flawless Flexbox Charts Grid
    html.Div([
        # Row 1 Charts: Q2 Revenue Trend + Q4 Delivery Delay Treemap
        html.Div([
            html.Div([dcc.Graph(id='chart-revenue-trend')], style={'flex': '1', 'minWidth': '48%', 'margin': '10px', 'backgroundColor': bg_card, 'border': f'1px solid {border_card}', 'borderRadius': '12px'}),
            html.Div([dcc.Graph(id='chart-delay-treemap')], style={'flex': '1', 'minWidth': '48%', 'margin': '10px', 'backgroundColor': bg_card, 'border': f'1px solid {border_card}', 'borderRadius': '12px'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'width': '100%'}),

        # Row 2 Charts: Q5 Regression Coefficients + Q3 Payment Distribution
        html.Div([
            html.Div([dcc.Graph(id='chart-regression-coef')], style={'flex': '1', 'minWidth': '48%', 'margin': '10px', 'backgroundColor': bg_card, 'border': f'1px solid {border_card}', 'borderRadius': '12px'}),
            html.Div([dcc.Graph(id='chart-installment-distribution')], style={'flex': '1', 'minWidth': '48%', 'margin': '10px', 'backgroundColor': bg_card, 'border': f'1px solid {border_card}', 'borderRadius': '12px'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'width': '100%'}),

        # Row 3 Charts: New Sweet Spot Analysis + New Raw Deal Scatter
        html.Div([
            html.Div([dcc.Graph(id='chart-sweet-spot')], style={'flex': '1', 'minWidth': '48%', 'margin': '10px', 'backgroundColor': bg_card, 'border': f'1px solid {border_card}', 'borderRadius': '12px'}),
            html.Div([dcc.Graph(id='chart-raw-deal-scatter')], style={'flex': '1', 'minWidth': '48%', 'margin': '10px', 'backgroundColor': bg_card, 'border': f'1px solid {border_card}', 'borderRadius': '12px'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'width': '100%'}),

        # Row 4 Table: New Operational Red Zone Table
        html.Div([
            html.Div([
                html.H3("🚨 THE OPERATIONAL RED ZONE: TOP DELAYED CITIES (MIN 10 ORDERS)", 
                        style={'fontSize': '15px', 'color': text_main, 'margin': '15px 0 10px 0', 'letterSpacing': '1px', 'fontWeight': 'bold', 'textAlign': 'center'}),
                html.Div(id='table-operational-red-zone', style={'padding': '0 20px 20px 20px', 'overflowX': 'auto'})
            ], style={'flex': '1', 'margin': '10px', 'backgroundColor': bg_card, 'border': f'1px solid {border_card}', 'borderRadius': '12px'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'width': '100%'}),

        # Row 5: Review Prediction Widget
        html.Div([
            html.Div([
                html.H3("📦 PREDICT REVIEW FROM SHIPPING TIME", 
                        style={'fontSize': '15px', 'color': text_main, 'margin': '15px 0 10px 0', 'letterSpacing': '1px', 'fontWeight': 'bold', 'textAlign': 'center'}),
                html.Div([
                    html.Label("Transit Time (Days):", style={'color': text_sec, 'marginBottom': '8px', 'display': 'block', 'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='transit-slider',
                        min=1, max=30, step=1, value=10,
                        marks={i: f'{i}d' for i in range(1, 31, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'padding': '10px 20px 5px 20px'}),
                html.Div(id='prediction-display', style={'textAlign': 'center', 'padding': '10px 15px 5px 15px'}),
                html.Div(id='model-stats', style={'textAlign': 'center', 'padding': '0 15px 15px 15px', 'color': text_sec, 'fontSize': '12px'})
            ], style={'flex': '1', 'margin': '10px', 'backgroundColor': bg_card, 'border': f'1px solid {border_card}', 'borderRadius': '12px'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'width': '100%'})
    ], style={'padding': '0 10px'})

], style={'backgroundColor': bg_main, 'minHeight': '100vh', 'fontFamily': 'system-ui, sans-serif', 'paddingBottom': '40px'})


# ── 3. Crash-Proof Callbacks ──

def get_filtered_dataset(categories, states):
    """Safely subset the database based on multicriteria filters."""
    data = df.copy()
    if categories:
        data = data[data['Product Category'].isin(categories)]
    if states:
        data = data[data['customer_state'].isin(states)]
    return data


@app.callback(
    [Output('kpi-revenue', 'children'),
     Output('kpi-orders', 'children'),
     Output('kpi-satisfaction', 'children'),
     Output('kpi-transit', 'children')],
    [Input('category-filter', 'value'),
     Input('state-filter', 'value')]
)
def update_kpis(selected_categories, selected_states):
    data = get_filtered_dataset(selected_categories, selected_states)
    
    if data.empty:
        return "R$ 0.00", "0", "⭐ N/A", "N/A"
        
    revenue = data['Price of Item'].sum()
    unique_orders = data['Order ID'].nunique()
    avg_rating = data['review_score'].mean()
    avg_transit = data['Transit Time Days'].dropna().mean()
    
    transit_label = f"{avg_transit:.1f} Days" if not np.isnan(avg_transit) else "N/A"
    
    return f"R$ {revenue:,.2f}", f"{unique_orders:,}", f"⭐ {avg_rating:.2f} / 5.0", transit_label


@app.callback(
    Output('chart-revenue-trend', 'figure'),
    [Input('category-filter', 'value'),
     Input('state-filter', 'value')]
)
def update_revenue_trend(selected_categories, selected_states):
    data = get_filtered_dataset(selected_categories, selected_states)
    
    # Render placeholder if data is empty
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No transactions found for selected filters.", showarrow=False, font=dict(color=text_sec, size=16))
        fig.update_layout(plot_bgcolor=bg_card, paper_bgcolor=bg_card)
        return fig

    monthly = data.groupby('YearMonth').agg(
        revenue=('Price of Item', 'sum'),
        orders=('Order ID', 'nunique')
    ).reset_index()
    monthly = monthly.sort_values('YearMonth')
    monthly['YearMonth_Str'] = monthly['YearMonth'].astype(str)

    fig = go.Figure()
    # Add Revenue bar
    fig.add_trace(go.Bar(
        x=monthly['YearMonth_Str'], y=monthly['revenue'],
        name='Revenue (BRL)', marker_color='#38bdf8', opacity=0.85
    ))
    # Add Orders line
    fig.add_trace(go.Scatter(
        x=monthly['YearMonth_Str'], y=monthly['orders'],
        name='Orders', mode='lines+markers', yaxis='y2',
        line=dict(color='#10b981', width=3), marker=dict(size=6)
    ))

    fig.update_layout(
        title=dict(text='Monthly Transaction and Revenue Performance Growth (Q2)', font=dict(color=text_main, size=15)),
        xaxis=dict(title=dict(text='Month', font=dict(color=text_sec)), tickfont=dict(color=text_sec), gridcolor='#334155', tickangle=45),
        yaxis=dict(title=dict(text='Total Revenue (BRL)', font=dict(color='#38bdf8')), tickfont=dict(color='#38bdf8'), gridcolor='#334155'),
        yaxis2=dict(title=dict(text='Volume of Orders', font=dict(color='#10b981')), tickfont=dict(color='#10b981'), overlaying='y', side='right'),
        paper_bgcolor=bg_card, plot_bgcolor=bg_card,
        margin=dict(l=50, r=50, t=50, b=50), height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color=text_sec)),
        hovermode='x unified'
    )
    return fig


@app.callback(
    Output('chart-delay-treemap', 'figure'),
    [Input('category-filter', 'value'),
     Input('state-filter', 'value')]
)
def update_delay_treemap(selected_categories, selected_states):
    data = get_filtered_dataset(selected_categories, selected_states)
    
    # Filter for valid delivery records
    del_data = data[data['Delivery Date'].notnull() & data['Estimated Delivery Date'].notnull()].copy()
    del_data['Delivery Delay Days'] = (del_data['Delivery Date'] - del_data['Estimated Delivery Date']).dt.days
    
    if del_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No delivered records available for selected filters.", showarrow=False, font=dict(color=text_sec, size=16))
        fig.update_layout(plot_bgcolor=bg_card, paper_bgcolor=bg_card)
        return fig

    # Aggregate at city level for treemap hierarchy
    city_log = del_data.groupby(['customer_state', 'customer_city']).agg(
        avg_delay=('Delivery Delay Days', 'mean'),
        revenue=('Price of Item', 'sum'),
        orders=('Order ID', 'nunique')
    ).round(1).reset_index()

    # Filter for cities with a meaningful transaction volume to prevent unreadable visual noise
    city_log = city_log[city_log['orders'] >= 150]

    fig = px.treemap(
        city_log,
        path=[px.Constant('Brazil'), 'customer_state', 'customer_city'],
        values='orders',
        color='avg_delay',
        color_continuous_scale='RdYlBu_r',
        title='Q4 — Delivery Delay Drill-Down: State → City<br><sup>Box size = order volume, Color = avg delay (days)</sup>',
        hover_data={'revenue': ':,.0f', 'orders': ':,'}
    )
    fig.update_traces(
        textinfo='label+value',
        hovertemplate='<b>%{label}</b> / %{parent}<br>'
                      'Avg Delay: %{color:.1f} days<br>'
                      'Revenue: R$ %{customdata[0]:,.0f}<br>'
                      'Orders: %{customdata[1]:,}<extra></extra>'
    )
    fig.update_layout(
        paper_bgcolor=bg_card,
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
        font=dict(color=text_main)
    )
    return fig


@app.callback(
    Output('chart-regression-coef', 'figure'),
    [Input('category-filter', 'value'),
     Input('state-filter', 'value')]
)
def update_regression_chart(selected_categories, selected_states):
    data = get_filtered_dataset(selected_categories, selected_states)
    
    # Filter rows containing valid dates and financial fields
    reg_df = data[data['review_score'].notnull() & data['Transit Time Days'].notnull() & data['Price of Item'].notnull()].copy()
    
    # Safety Check: Regression needs at least 15 valid observations to run reliably on filtered data!
    if len(reg_df) < 15:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data scale to execute live Regression (Min: 15).", showarrow=False, font=dict(color=text_sec, size=15))
        fig.update_layout(plot_bgcolor=bg_card, paper_bgcolor=bg_card)
        return fig

    features = ['Price of Item', 'Freight Cost', 'Transit Time Days']
    X = reg_df[features]
    y = reg_df['review_score']

    try:
        # Standardize variables to compare coefficients on equal scales
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        coef_df = pd.DataFrame({
            'Feature': ['Product Price', 'Shipping Fee', 'Delivery Delay'],
            'Coeff': model.coef_
        }).sort_values(by='Coeff', key=abs, ascending=True)

        colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_df['Coeff']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=coef_df['Feature'], x=coef_df['Coeff'],
            orientation='h', marker_color=colors,
            text=coef_df['Coeff'].round(3), textposition='outside'
        ))
        
        fig.update_layout(
            title=dict(text='Interactive Real-Time Regression Drivers of Customer Reviews (Q5)', font=dict(color=text_main, size=15)),
            xaxis=dict(title=dict(text='Standardized Impact Scale', font=dict(color=text_sec)), tickfont=dict(color=text_sec), gridcolor='#334155'),
            yaxis=dict(tickfont=dict(color=text_main, size=12)),
            paper_bgcolor=bg_card, plot_bgcolor=bg_card,
            margin=dict(l=120, r=40, t=50, b=40), height=380
        )
        fig.add_vline(x=0, line_color='white', line_width=1)
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Live Regression execution error: {str(e)}", showarrow=False, font=dict(color=text_sec, size=14))
        fig.update_layout(plot_bgcolor=bg_card, paper_bgcolor=bg_card)
        return fig


@app.callback(
    [Output('prediction-display', 'children'),
     Output('model-stats', 'children')],
    [Input('category-filter', 'value'),
     Input('state-filter', 'value'),
     Input('transit-slider', 'value')]
)
def update_prediction(selected_categories, selected_states, transit_days):
    data = get_filtered_dataset(selected_categories, selected_states)

    reg_df = data[data['review_score'].notnull() & data['Transit Time Days'].notnull()].copy()

    if len(reg_df) < 15:
        empty_pred = html.Div("⚠️ Insufficient data (min 15 records required)", style={'color': '#f87171', 'fontSize': '14px'})
        return empty_pred, html.Div()

    try:
        X = reg_df[['Transit Time Days']]
        y = reg_df['review_score']

        model = LinearRegression()
        model.fit(X, y)

        predicted = model.predict([[transit_days]])[0]
        predicted = np.clip(predicted, 1, 5)
        r2 = model.score(X, y)
        coef = model.coef_[0]
        intercept = model.intercept_

        stars_count = max(1, round(predicted))
        stars = '⭐' * stars_count

        pred_html = html.Div([
            html.Div(stars, style={'fontSize': '28px', 'margin': '5px 0'}),
            html.Div(f'Predicted: {predicted:.2f} / 5.0', 
                     style={'fontSize': '22px', 'fontWeight': 'bold', 'color': accent_color, 'margin': '8px 0'}),
            html.Div(f'for {transit_days} day{"s" if transit_days != 1 else ""} transit',
                     style={'fontSize': '14px', 'color': text_sec}),
        ])

        stats_html = html.Div([
            html.Hr(style={'borderColor': border_card, 'margin': '10px 0'}),
            html.Div(f'Review = {intercept:.3f} {"+" if coef >= 0 else ""} ({coef:.4f} × Transit Days)',
                     style={'margin': '4px 0'}),
            html.Div(f'R² = {r2:.3f} | Each shipping day {"raises" if coef >= 0 else "lowers"} review by {abs(coef):.4f} pts',
                     style={'margin': '4px 0'}),
            html.Div(f'Trained on {len(reg_df):,} orders', style={'margin': '4px 0', 'fontSize': '11px'}),
        ])

        return pred_html, stats_html

    except Exception as e:
        error_pred = html.Div(f"⚠️ Model error: {str(e)}", style={'color': '#f87171', 'fontSize': '14px'})
        return error_pred, html.Div()


@app.callback(
    Output('chart-installment-distribution', 'figure'),
    [Input('category-filter', 'value'),
     Input('state-filter', 'value')]
)
def update_installment_chart(selected_categories, selected_states):
    data = get_filtered_dataset(selected_categories, selected_states)
    
    # Exclude Unknown categories or null payments
    clean_data = data[(data['Product Category'] != 'Unknown') & data['Payment Type'].notnull()]
    
    if clean_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No payment records found.", showarrow=False, font=dict(color=text_sec, size=16))
        fig.update_layout(plot_bgcolor=bg_card, paper_bgcolor=bg_card)
        return fig

    # Focus on top 10 categories in the current active subset
    top_cats = clean_data['Product Category'].value_counts().head(10).index
    data_top = clean_data[clean_data['Product Category'].isin(top_cats)]

    if data_top.empty:
        fig = go.Figure()
        fig.add_annotation(text="No transactional data in top categories.", showarrow=False, font=dict(color=text_sec, size=16))
        fig.update_layout(plot_bgcolor=bg_card, paper_bgcolor=bg_card)
        return fig

    # Build cross-tabulation table
    ct = pd.crosstab(
        data_top['Product Category'],
        data_top['Payment Type'],
        normalize='index'
    ) * 100

    # SAFETY CHECK: If 'credit_card' column is not in columns, sort by whatever is actually available!
    sort_col = 'credit_card' if 'credit_card' in ct.columns else ct.columns[0]
    ct = ct.sort_values(sort_col, ascending=True)
    ct = ct.round(1)

    fig = go.Figure()
    
    # Dynamic palette mapping for payments
    pmt_colors = {
        'credit_card': '#38bdf8',  # Sky 400
        'boleto': '#f59e0b',       # Amber 500
        'voucher': '#ec4899',      # Pink 500
        'debit_card': '#10b981'    # Emerald 500
    }

    for col in ct.columns:
        display_name = col.replace('_', ' ').title()
        color = pmt_colors.get(col, '#94a3b8')
        fig.add_trace(go.Bar(
            name=display_name,
            y=ct.index,
            x=ct[col],
            orientation='h',
            marker_color=color,
            text=ct[col].apply(lambda x: f'{x:.0f}%' if x > 5 else ''),
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=9, color='#000000' if color in ['#f59e0b', '#38bdf8'] else '#ffffff')
        ))

    fig.update_layout(
        title=dict(text='Checkout Payment Gateway Split by Category (Q3)', font=dict(color=text_main, size=15)),
        xaxis=dict(title=dict(text='% of transactions within Category', font=dict(color=text_sec)), tickfont=dict(color=text_sec), gridcolor='#334155'),
        yaxis=dict(tickfont=dict(color=text_main, size=11)),
        barmode='stack',
        paper_bgcolor=bg_card, plot_bgcolor=bg_card,
        margin=dict(l=140, r=40, t=50, b=40), height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color=text_sec)),
        hovermode='y unified'
    )
    return fig


@app.callback(
    Output('chart-sweet-spot', 'figure'),
    [Input('category-filter', 'value'),
     Input('state-filter', 'value')]
)
def update_sweet_spot_scatter(selected_categories, selected_states):
    data = get_filtered_dataset(selected_categories, selected_states)
    clean_data = data[data['Product Category'] != 'Unknown']
    
    if clean_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No product category records found.", showarrow=False, font=dict(color=text_sec, size=16))
        fig.update_layout(plot_bgcolor=bg_card, paper_bgcolor=bg_card)
        return fig

    # Group by category and compute stats
    cat_stats = clean_data.groupby('Product Category').agg(
        avg_price=('Price of Item', 'mean'),
        pct_5_star=('review_score', lambda x: (x == 5.0).mean() * 100),
        volume=('Order ID', 'count')
    ).reset_index()

    # Filter to categories with at least 150 orders to avoid small-sample noise
    cat_stats = cat_stats[cat_stats['volume'] >= 150]

    if cat_stats.empty:
        fig = go.Figure()
        fig.add_annotation(text="No product categories met the 150+ orders volume threshold.", showarrow=False, font=dict(color=text_sec, size=16))
        fig.update_layout(plot_bgcolor=bg_card, paper_bgcolor=bg_card)
        return fig

    fig = px.scatter(
        cat_stats,
        x='avg_price',
        y='pct_5_star',
        size='volume',
        color='pct_5_star',
        color_continuous_scale='RdYlGn',
        title='Q6 — Sweet Spot Analysis: High Price + High Satisfaction Categories<br><sup>Bubble size = order volume, Color = % 5-star reviews</sup>',
        labels={'avg_price': 'Average Item Price (BRL)', 'pct_5_star': '% 5-Star Reviews', 'Product Category': 'Category'},
        hover_data={'Product Category': True, 'avg_price': ':,.2f', 'pct_5_star': ':.2f', 'volume': ':,'}
    )

    fig.update_traces(
        marker=dict(sizemin=5),
        hovertemplate='<b>%{customdata[0]}</b><br>'
                      'Avg Price: R$ %{x:,.2f}<br>'
                      '5-Star Reviews: %{y:.2f}%%<br>'
                      'Volume: %{marker.size:,}<extra></extra>'
    )

    # Dark theme layout adjustments
    fig.update_layout(
        paper_bgcolor=bg_card,
        plot_bgcolor=bg_card,
        margin=dict(l=50, r=50, t=70, b=50),
        height=380,
        font=dict(color=text_main),
        coloraxis_colorbar=dict(title=dict(text='% 5-Star', font=dict(color=text_sec)), tickfont=dict(color=text_sec))
    )

    fig.update_xaxes(gridcolor='#334155', tickfont=dict(color=text_sec))
    fig.update_yaxes(gridcolor='#334155', tickfont=dict(color=text_sec))

    # Reference median lines
    median_pct = cat_stats['pct_5_star'].median()
    median_price = cat_stats['avg_price'].median()

    fig.add_hline(
        y=median_pct, line_dash='dash', line_color='#94a3b8',
        annotation_text=f'Median Satisfaction ({median_pct:.1f}%)',
        annotation_position='top left',
        annotation_font=dict(color=text_sec, size=10)
    )
    fig.add_vline(
        x=median_price, line_dash='dash', line_color='#94a3b8',
        annotation_text=f'Median Price (R$ {median_price:.1f})',
        annotation_position='top right',
        annotation_font=dict(color=text_sec, size=10)
    )

    return fig


@app.callback(
    Output('chart-raw-deal-scatter', 'figure'),
    [Input('category-filter', 'value'),
     Input('state-filter', 'value')]
)
def update_raw_deal_scatter(selected_categories, selected_states):
    data = get_filtered_dataset(selected_categories, selected_states)
    
    del_data = data[data['Delivery Date'].notnull() & data['Estimated Delivery Date'].notnull()].copy()
    del_data['Delivery Delay Days'] = (del_data['Delivery Date'] - del_data['Estimated Delivery Date']).dt.days
    
    if del_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No delivered records available for selected filters.", showarrow=False, font=dict(color=text_sec, size=16))
        fig.update_layout(plot_bgcolor=bg_card, paper_bgcolor=bg_card)
        return fig

    state_log = del_data.groupby('customer_state').agg(
        avg_freight=('Freight Cost', 'mean'),
        avg_delay=('Delivery Delay Days', 'mean'),
        orders=('Order ID', 'nunique')
    ).round(2).reset_index()

    fig = px.scatter(
        state_log,
        x='avg_freight',
        y='avg_delay',
        size='orders',
        color='avg_delay',
        color_continuous_scale='RdYlBu_r',
        text='customer_state',
        title='Q7 — Ad Hoc: Geographic Shipping Fee vs. Average Delay by State<br><sup>Bubble size = order volume, Color = average delay (days)</sup>',
        labels={'avg_freight': 'Average Freight Cost (BRL)', 'avg_delay': 'Average Delivery Delay (Days)', 'customer_state': 'State'},
        hover_data={'avg_freight': ':,.2f', 'avg_delay': ':.2f', 'orders': ':,'}
    )

    fig.update_traces(
        textposition='top center',
        marker=dict(sizemin=5),
        hovertemplate='<b>%{text}</b><br>'
                      'Avg Freight: R$ %{x:,.2f}<br>'
                      'Avg Delay: %{y:.1f} days<br>'
                      'Orders: %{marker.size:,}<extra></extra>'
    )
    fig.update_layout(
        paper_bgcolor=bg_card,
        plot_bgcolor=bg_card,
        margin=dict(l=50, r=50, t=60, b=50),
        height=380,
        font=dict(color=text_main),
        coloraxis_colorbar=dict(title=dict(text='Avg Delay', font=dict(color=text_sec)), tickfont=dict(color=text_sec))
    )
    
    fig.update_xaxes(gridcolor='#334155', tickfont=dict(color=text_sec))
    fig.update_yaxes(gridcolor='#334155', tickfont=dict(color=text_sec))
    
    fig.add_hline(
        y=0, line_dash='dash', line_color='#94a3b8',
        annotation_text='On-time delivery threshold',
        annotation_position='bottom right',
        annotation_font=dict(color=text_sec, size=10)
    )
    return fig


@app.callback(
    Output('table-operational-red-zone', 'children'),
    [Input('category-filter', 'value'),
     Input('state-filter', 'value')]
)
def update_red_zone_table(selected_categories, selected_states):
    data = get_filtered_dataset(selected_categories, selected_states)
    
    del_data = data[data['Delivery Date'].notnull() & data['Estimated Delivery Date'].notnull()].copy()
    del_data['Delivery Delay Days'] = (del_data['Delivery Date'] - del_data['Estimated Delivery Date']).dt.days
    
    if del_data.empty:
        return html.Div("No delivered records available.", style={'textAlign': 'center', 'color': text_sec, 'padding': '20px'})

    city_log = del_data.groupby(['customer_state', 'customer_city']).agg(
        avg_delay=('Delivery Delay Days', 'mean'),
        orders=('Order ID', 'nunique'),
        avg_rating=('review_score', 'mean')
    ).reset_index()

    red_zone = city_log[(city_log['orders'] >= 10) & (city_log['avg_delay'] > 0)]
    
    if red_zone.empty:
        return html.Div("🎉 Excellent! All high-volume cities in selected filters average on-time or early deliveries (Zero positive delays).", 
                        style={'textAlign': 'center', 'color': '#10b981', 'padding': '20px', 'fontWeight': 'bold'})

    top_red = red_zone.sort_values(by='avg_delay', ascending=False).head(5)
    top_red['avg_delay'] = top_red['avg_delay'].round(1)
    top_red['avg_rating'] = top_red['avg_rating'].round(2)

    table_header = html.Thead(
        html.Tr([
            html.Th("City", style={'borderBottom': '2px solid #475569', 'padding': '12px', 'textAlign': 'left', 'color': text_main, 'backgroundColor': '#334155'}),
            html.Th("State", style={'borderBottom': '2px solid #475569', 'padding': '12px', 'textAlign': 'center', 'color': text_main, 'backgroundColor': '#334155'}),
            html.Th("Total Orders", style={'borderBottom': '2px solid #475569', 'padding': '12px', 'textAlign': 'center', 'color': text_main, 'backgroundColor': '#334155'}),
            html.Th("Average Delay", style={'borderBottom': '2px solid #475569', 'padding': '12px', 'textAlign': 'center', 'color': text_main, 'backgroundColor': '#334155'}),
            html.Th("Avg Review Score", style={'borderBottom': '2px solid #475569', 'padding': '12px', 'textAlign': 'center', 'color': text_main, 'backgroundColor': '#334155'})
        ])
    )

    table_rows = []
    for _, row in top_red.iterrows():
        table_rows.append(
            html.Tr([
                html.Td(row['customer_city'].title(), style={'padding': '12px', 'textAlign': 'left', 'borderBottom': '1px solid #334155'}),
                html.Td(row['customer_state'], style={'padding': '12px', 'textAlign': 'center', 'borderBottom': '1px solid #334155', 'fontWeight': 'bold', 'color': accent_color}),
                html.Td(f"{row['orders']:,}", style={'padding': '12px', 'textAlign': 'center', 'borderBottom': '1px solid #334155'}),
                html.Td(f"+{row['avg_delay']} Days", style={'padding': '12px', 'textAlign': 'center', 'borderBottom': '1px solid #334155', 'color': '#f87171', 'fontWeight': 'bold'}),
                html.Td(f"⭐ {row['avg_rating']:.2f}", style={'padding': '12px', 'textAlign': 'center', 'borderBottom': '1px solid #334155', 'color': '#fbbf24', 'fontWeight': 'bold'})
            ])
        )

    table_body = html.Tbody(table_rows)

    return html.Table(
        [table_header, table_body],
        style={'width': '100%', 'borderCollapse': 'collapse', 'backgroundColor': bg_card, 'color': text_main, 'borderRadius': '8px', 'overflow': 'hidden'}
    )


# ── 4. Run App ──
if __name__ == '__main__':
    # Run dashboard locally on port 8050
    app.run(debug=False, port=8050, host='127.0.0.1')
