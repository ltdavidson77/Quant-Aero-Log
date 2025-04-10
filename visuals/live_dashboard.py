# ==========================
# visuals/live_dashboard.py
# ==========================
# Live streaming dashboard for signals (Bokeh).

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, HoverTool, Range1d
from bokeh.layouts import column, row
from bokeh.palettes import Category10
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio

class LiveDashboard:
    def __init__(self, 
                 window_size: int = 200,
                 update_interval: int = 1000,
                 title: str = "Live Price & Signal Tracker"):
        self.window_size = window_size
        self.update_interval = update_interval
        self.title = title
        
        # Initialize data source with empty arrays
        self.data_source = ColumnDataSource(data={
            'time': [],
            'price': [],
            'signal': [],
            'volume': []
        })
        
        # Create main plot
        self.plot = figure(
            title=title,
            x_axis_type="datetime",
            plot_height=400,
            plot_width=1000,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_drag='pan',
            active_scroll='wheel_zoom'
        )
        
        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Time", "@time{%Y-%m-%d %H:%M:%S}"),
                ("Price", "@price{0.2f}"),
                ("Signal", "@signal{0.2f}"),
                ("Volume", "@volume{0.2f}")
            ],
            formatters={
                '@time': 'datetime'
            }
        )
        self.plot.add_tools(hover)
        
        # Configure plot appearance
        self.plot.xaxis.axis_label = "Time"
        self.plot.yaxis.axis_label = "Value"
        self.plot.grid.grid_line_alpha = 0.3
        self.plot.legend.location = "top_left"
        self.plot.legend.click_policy = "hide"
        
        # Add lines with different colors
        self.plot.line('time', 'price', source=self.data_source, 
                      legend_label="Price", color=Category10[3][0], line_width=2)
        self.plot.line('time', 'signal', source=self.data_source, 
                      legend_label="Signal", color=Category10[3][1], line_width=2)
        
        # Add volume bars
        self.plot.vbar(x='time', top='volume', source=self.data_source,
                      width=timedelta(milliseconds=500), alpha=0.5,
                      color=Category10[3][2])
    
    def update(self) -> None:
        """Update the dashboard with new data."""
        new_time = datetime.now()
        new_price = 100 + np.random.normal(0, 0.5)
        new_signal = 50 + np.random.normal(0, 2)
        new_volume = np.random.uniform(0, 10)
        
        new_data = {
            'time': [new_time],
            'price': [new_price],
            'signal': [new_signal],
            'volume': [new_volume]
        }
        
        # Stream new data with rollover
        self.data_source.stream(new_data, rollover=self.window_size)
        
        # Auto-scale y-axis
        y_min = min(min(self.data_source.data['price']), 
                   min(self.data_source.data['signal'])) * 0.95
        y_max = max(max(self.data_source.data['price']), 
                   max(self.data_source.data['signal'])) * 1.05
        self.plot.y_range = Range1d(y_min, y_max)
    
    def start(self) -> None:
        """Start the dashboard server."""
        curdoc().add_root(column(self.plot))
        curdoc().add_periodic_callback(self.update, self.update_interval)

# Create and start dashboard
dashboard = LiveDashboard()
dashboard.start()
