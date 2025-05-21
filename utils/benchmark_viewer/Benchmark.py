from plotly.graph_objs import YAxis
import streamlit as st
import random
import pandas as pd
import plotly.express as px
import os
import json

# Cache the data extraction function
@st.cache_data
def extract_f1_scores(base_path):
    data = []
    for root, _, files in os.walk(base_path):
        if "metrics.json" in files:
            print("Found metrics.json in", root)
            model_name = os.path.basename(root)
            # Get the category path from the folder structure
            folder_parts = os.path.relpath(root, base_path).split(os.sep)
            category = '/'.join(folder_parts[:-1])  # Combine parent folders as category
            
            file_path = os.path.join(root, "metrics.json")
            with open(file_path, "r") as f:
                try:
                    metrics = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading {file_path}")
                    continue
                f1_score = metrics.get("macro avg", {}).get("f1-score")
                if f1_score is not None:
                    data.append({"Category": category, "Model": model_name, "F1 Score": f1_score})
    return pd.DataFrame(data)

# Cache the hierarchy creation function
@st.cache_data
def get_hierarchy(df):
    df['Hierarchy'] = df['Category'].apply(lambda x: x.split('/'))
    return df

# Streamlit app setup
st.title("Model Benchmark Chart")

# Add option to change the results folder
results_path = st.text_input("Enter the path to the results folder", value="results")

# Check if the folder exists
if not os.path.exists(results_path):
    st.error(f"The folder '{results_path}' does not exist. Please enter a valid path.")
    st.stop()

# Extract and cache the data
df = extract_f1_scores(results_path)

# If no data was found
if df.empty:
    st.write("No data found in the results folder.")
else:
    # Add hierarchy to the DataFrame
    df = get_hierarchy(df)

    # Function to plot the F1 scores with sorting
    def plot_f1_score(filtered_df, category):
        # Extract the last-level category name for the title
        last_category_name = category.split('/')[-1].replace('_', ' ').capitalize()
        
        # Modify the 'Category' column in the filtered_df to use only the last part of the path
        filtered_df = filtered_df.copy()
        filtered_df['Category'] = filtered_df['Category'].apply(lambda x: x.split('/')[-1].replace('_', ' ').capitalize())
        
        # Add sorting options
        sort_option = st.selectbox(
            "Sort by",
            options=["F1 Score (High to Low)", "F1 Score (Low to High)", "Model (A-Z)", "Model (Z-A)"],
            key=f"sort_option_{category}_{last_category_name}"  # Unique key for each category
        )
        
        # Apply sorting based on user selection
        if sort_option == "Model (A-Z)":
            filtered_df = filtered_df.sort_values(by="Model", ascending=True)
        elif sort_option == "Model (Z-A)":
            filtered_df = filtered_df.sort_values(by="Model", ascending=False)
        elif sort_option == "F1 Score (High to Low)":
            filtered_df = filtered_df.sort_values(by="F1 Score", ascending=True)
        elif sort_option == "F1 Score (Low to High)":
            filtered_df = filtered_df.sort_values(by="F1 Score", ascending=False)
        
        # Plot using the modified 'Category' column
        fig = px.bar(filtered_df, 
                     y="Model", 
                     x="F1 Score", 
                     orientation="h",
                     color="Category",  # Use the modified 'Category' column for coloring
                     barmode="group",
                     text="F1 Score",
                     title=f"Models in {last_category_name} (Sorted by {sort_option})")  # Include sorting info in the title

        fig.update_layout(xaxis_range=[0, 1], xaxis_title="Macro Avg F1", yaxis_title="Model")
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        return fig

    # Modified render_hierarchy function with stable expander states
    def render_hierarchy(df, category_path, level=0):
        filtered_df = df[df['Hierarchy'].apply(lambda x: len(x) > level and x[:level + 1] == category_path)]
        
        if filtered_df.empty:
            return
            
        next_level_categories = filtered_df['Hierarchy'].apply(lambda x: x[level + 1] if len(x) > level + 1 else None).dropna().unique()
        section_title = category_path[level].replace('_', ' ').capitalize()

        # Check if this is the last level in the hierarchy, account for when the hierarchy is shorter than 2 levels
        is_last_level = all(len(x) <= level + 2 for x in filtered_df['Hierarchy']) 

        if level == 1:
            # Stable expander key based on full category path
            expander_key = f"expander_{'_'.join(category_path)}"
            
            # Initialize state only once
            if expander_key not in st.session_state:
                st.session_state[expander_key] = False
                
            # Create expander with stable state
            with st.expander(section_title, expanded=st.session_state[expander_key]):
                _render_content(filtered_df, category_path, level, is_last_level, next_level_categories)
        else:
            st.markdown(f"{'#' * level} {section_title}")
            _render_content(filtered_df, category_path, level, is_last_level, next_level_categories)

    # Helper function to render content (to avoid code duplication)
    def _render_content(filtered_df, category_path, level, is_last_level, next_level_categories):
        # If this is the second-lowest level, stop recursion and plot all leaf nodes
        if is_last_level:
            # Determine the depth of the current level relative to metrics.json
            if level == 0:
                # If we're at the top level, filter models at level 1
                leaf_nodes_df = filtered_df[filtered_df['Hierarchy'].apply(lambda x: len(x) == 1)]
            else:
                # Otherwise, filter models at level + 2
                leaf_nodes_df = filtered_df[filtered_df['Hierarchy'].apply(lambda x: len(x) == level + 2)]
                leaf_nodes_df = filtered_df[filtered_df['Hierarchy'].apply(lambda x: len(x) == level + 2)]

            
            # Menu for models visibility at this level
            models = leaf_nodes_df['Model'].unique()
            
            # Retrieve or initialize the session state for selected models
            model_key = f"models_{'_'.join(category_path)}"
            if model_key not in st.session_state:
                st.session_state[model_key] = models.tolist()

            show_models = st.multiselect("Select models to display", options=models, key=model_key, default=st.session_state[model_key])

            # Filter data to show selected models only
            filtered_leaf_nodes_df = leaf_nodes_df[leaf_nodes_df['Model'].isin(show_models)]

            # Plot the F1 scores for the models
            if not filtered_leaf_nodes_df.empty:
                fig = plot_f1_score(filtered_leaf_nodes_df, '/'.join(category_path))
                st.plotly_chart(fig)
        else:
            # Recurse through the next level categories
            for next_category in next_level_categories:
                render_hierarchy(df, category_path + [next_category], level + 1)

    # Get top-level categories (first level in the hierarchy)
    top_categories = df['Hierarchy'].apply(lambda x: x[0]).unique()

    # Category selection dropdown
    selected_category = st.multiselect(
        "Select category to display",
        options=top_categories,
        key="category_selector",
        default=top_categories,
    )

    # Start rendering the hierarchical structure for selected category
    for category in selected_category:
        render_hierarchy(df, [category], level=0)


    # Download button for the benchmark data
    st.download_button("Download Benchmark Data", df.to_csv(index=False), "benchmark_results.csv", "text/csv")

