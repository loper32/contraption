"""
Correlation and relationship visualization tools
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class CorrelationVisualizer:
    """
    Creates visualizations for metric correlations and relationships.
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize the correlation visualizer.

        Args:
            theme: Plotly theme to use for plots
        """
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set3

    def create_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Metric Correlations",
        width: int = 800,
        height: int = 600
    ) -> go.Figure:
        """
        Create an interactive correlation heatmap.

        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Plot title
            width: Plot width
            height: Plot height

        Returns:
            Plotly figure object
        """
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(correlation_matrix.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(
                title="Correlation"
            )
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Metrics",
            yaxis_title="Metrics",
            width=width,
            height=height,
            template=self.theme
        )

        logger.info(f"Created correlation heatmap for {len(correlation_matrix.columns)} metrics")

        return fig

    def create_relationship_network(
        self,
        strong_relationships: List[Dict],
        threshold: float = 0.7
    ) -> go.Figure:
        """
        Create a network graph showing strong relationships between metrics.

        Args:
            strong_relationships: List of relationship dictionaries
            threshold: Minimum correlation to display

        Returns:
            Plotly figure object
        """
        # Filter relationships by threshold
        filtered_relationships = [
            rel for rel in strong_relationships
            if abs(rel['correlation']) >= threshold
        ]

        if not filtered_relationships:
            # Return empty plot
            fig = go.Figure()
            fig.add_annotation(
                text=f"No relationships found above threshold {threshold}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Extract unique metrics
        metrics = set()
        for rel in filtered_relationships:
            metrics.add(rel['metric1'])
            metrics.add(rel['metric2'])
        metrics = list(metrics)

        # Create node positions (circular layout)
        n_metrics = len(metrics)
        angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False)
        node_positions = {
            metric: (np.cos(angle), np.sin(angle))
            for metric, angle in zip(metrics, angles)
        }

        # Create edges
        edge_x = []
        edge_y = []
        edge_info = []

        for rel in filtered_relationships:
            x0, y0 = node_positions[rel['metric1']]
            x1, y1 = node_positions[rel['metric2']]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{rel['metric1']} ↔ {rel['metric2']}: {rel['correlation']:.3f}")

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Create node trace
        node_x = [node_positions[metric][0] for metric in metrics]
        node_y = [node_positions[metric][1] for metric in metrics]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=metrics,
            textposition="middle center",
            marker=dict(
                size=50,
                color='lightblue',
                line=dict(width=2, color='rgb(50,50,50)')
            )
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Metric Relationship Network (|r| ≥ {threshold})',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hover over nodes and edges for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))

        logger.info(f"Created relationship network with {len(metrics)} metrics and {len(filtered_relationships)} connections")

        return fig

    def create_lag_correlation_plot(
        self,
        lag_results: Dict,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create a plot showing lag correlations between two metrics.

        Args:
            lag_results: Results from lag correlation analysis
            title: Custom plot title

        Returns:
            Plotly figure object
        """
        if title is None:
            title = f"Lag Correlations: {lag_results['metric1']} vs {lag_results['metric2']}"

        # Create bar plot
        fig = go.Figure(data=go.Bar(
            x=lag_results['lags'],
            y=lag_results['correlations'],
            marker_color=['red' if corr < 0 else 'blue' for corr in lag_results['correlations']]
        ))

        # Highlight optimal lag
        optimal_idx = lag_results['lags'].index(lag_results['optimal_lag'])
        fig.add_shape(
            type="rect",
            x0=lag_results['optimal_lag'] - 0.4,
            y0=min(lag_results['correlations']) - 0.1,
            x1=lag_results['optimal_lag'] + 0.4,
            y1=max(lag_results['correlations']) + 0.1,
            line=dict(color="gold", width=3),
            fillcolor="rgba(255, 215, 0, 0.2)"
        )

        fig.update_layout(
            title=title,
            xaxis_title="Lag (periods)",
            yaxis_title="Correlation",
            template=self.theme,
            annotations=[
                dict(
                    x=lag_results['optimal_lag'],
                    y=lag_results['optimal_correlation'],
                    text=f"Optimal: {lag_results['optimal_lag']} periods<br>r = {lag_results['optimal_correlation']:.3f}",
                    showarrow=True,
                    arrowhead=2,
                    bgcolor="white",
                    bordercolor="gold",
                    borderwidth=2
                )
            ]
        )

        logger.info(f"Created lag correlation plot for {lag_results['metric1']} vs {lag_results['metric2']}")

        return fig

    def create_scatter_matrix(
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        color_metric: Optional[str] = None
    ) -> go.Figure:
        """
        Create a scatter plot matrix for multiple metrics.

        Args:
            df: Input DataFrame
            metrics: List of metrics to include
            color_metric: Metric to use for color coding

        Returns:
            Plotly figure object
        """
        if metrics is None:
            metrics = df.select_dtypes(include=[np.number]).columns.tolist()[:5]

        if len(metrics) > 6:
            logger.warning("Too many metrics for scatter matrix. Using first 6.")
            metrics = metrics[:6]

        # Create scatter matrix
        fig = px.scatter_matrix(
            df,
            dimensions=metrics,
            color=color_metric,
            title="Metrics Scatter Matrix",
            template=self.theme
        )

        fig.update_traces(diagonal_visible=False)
        fig.update_layout(
            height=600,
            width=800
        )

        logger.info(f"Created scatter matrix for {len(metrics)} metrics")

        return fig

    def create_time_series_comparison(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        title: str = "Time Series Comparison"
    ) -> go.Figure:
        """
        Create a time series plot comparing multiple metrics.

        Args:
            df: DataFrame with datetime index
            metrics: List of metrics to plot
            title: Plot title

        Returns:
            Plotly figure object
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index for time series plot")

        # Create subplots for each metric
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            shared_xaxes=True,
            subplot_titles=metrics,
            vertical_spacing=0.05
        )

        colors = px.colors.qualitative.Set1

        for i, metric in enumerate(metrics):
            if metric in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[metric],
                        name=metric,
                        line=dict(color=colors[i % len(colors)]),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )

        fig.update_layout(
            title=title,
            height=150 * len(metrics),
            template=self.theme
        )

        fig.update_xaxes(title_text="Date", row=len(metrics), col=1)

        logger.info(f"Created time series comparison for {len(metrics)} metrics")

        return fig

    def create_distribution_comparison(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        plot_type: str = "histogram"
    ) -> go.Figure:
        """
        Create distribution plots for comparing metrics.

        Args:
            df: Input DataFrame
            metrics: List of metrics to compare
            plot_type: Type of plot ('histogram', 'box', 'violin')

        Returns:
            Plotly figure object
        """
        if plot_type == "histogram":
            fig = make_subplots(
                rows=len(metrics),
                cols=1,
                subplot_titles=metrics,
                vertical_spacing=0.08
            )

            for i, metric in enumerate(metrics):
                if metric in df.columns:
                    fig.add_trace(
                        go.Histogram(
                            x=df[metric].dropna(),
                            name=metric,
                            showlegend=False,
                            opacity=0.7
                        ),
                        row=i+1, col=1
                    )

        elif plot_type == "box":
            # Melt the dataframe for box plots
            plot_data = df[metrics].melt(var_name='Metric', value_name='Value')

            fig = px.box(
                plot_data,
                x='Metric',
                y='Value',
                title="Distribution Comparison (Box Plots)"
            )

        elif plot_type == "violin":
            # Melt the dataframe for violin plots
            plot_data = df[metrics].melt(var_name='Metric', value_name='Value')

            fig = px.violin(
                plot_data,
                x='Metric',
                y='Value',
                title="Distribution Comparison (Violin Plots)"
            )

        fig.update_layout(
            title=f"Distribution Comparison ({plot_type.title()})",
            template=self.theme
        )

        logger.info(f"Created {plot_type} distribution comparison for {len(metrics)} metrics")

        return fig

    def create_correlation_strength_chart(
        self,
        strong_relationships: List[Dict],
        top_n: int = 10
    ) -> go.Figure:
        """
        Create a horizontal bar chart showing correlation strengths.

        Args:
            strong_relationships: List of relationship dictionaries
            top_n: Number of top relationships to show

        Returns:
            Plotly figure object
        """
        # Sort and take top N
        sorted_relationships = sorted(
            strong_relationships,
            key=lambda x: abs(x['correlation']),
            reverse=True
        )[:top_n]

        # Prepare data
        relationship_labels = [
            f"{rel['metric1']} ↔ {rel['metric2']}"
            for rel in sorted_relationships
        ]
        correlations = [rel['correlation'] for rel in sorted_relationships]
        colors = ['red' if corr < 0 else 'blue' for corr in correlations]

        # Create horizontal bar chart
        fig = go.Figure(data=go.Bar(
            y=relationship_labels,
            x=correlations,
            orientation='h',
            marker_color=colors,
            text=[f"{corr:.3f}" for corr in correlations],
            textposition='auto'
        ))

        fig.update_layout(
            title=f"Top {top_n} Strongest Correlations",
            xaxis_title="Correlation Coefficient",
            yaxis_title="Metric Pairs",
            template=self.theme,
            height=max(400, 30 * len(sorted_relationships))
        )

        logger.info(f"Created correlation strength chart for top {len(sorted_relationships)} relationships")

        return fig