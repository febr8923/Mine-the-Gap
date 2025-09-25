# Data for the Challenge "Mine the Gap"

### How can we automatically detect anomalies in a live, evolving knowledge graph?

Welcome to our challenge!

At Swisscom we continuously collect real-time information about our network infrastructure. All of this is stored in a massive knowledge graph with more than 200 million nodes, capturing dependencies between network devices and services.

For this challenge you’ll receive a synthetic, simplified dataset that mimics our real infrastructure. Although we clean the data as much as possible, anomalies sometimes appear in the graph. Spotting them isn't a trivial task, and that’s the goal of this challenge. Try to design a method that can detect them automatically.

Anomalies can occur at different scales (node, edge, subgraph), but here we focus on edge-level anomalies.

In the data repository you’ll find two datasets. Both record edge events in our graph (think of them as logs): whenever an edge is added or removed, we store that event. The file <code>edge_events_clean.csv</code> contains a graph without anomalies; the other file contains a graph with the same overall structure but with introduced anomalies (some edges removed, some unexpected edges added). Use them as you see fit.

Any approach to crack this problem is welcome! One promising direction is to use Graph Neural Networks (GNNs), since they can learn representations for nodes and edges and help identify unusual patterns. In the notebook <code>tutorial.ipynb</code> we show how to load the data into PyTorch Geometric format, the de-facto standard library for Graph ML, but feel free to use any other tools or frameworks you prefer!

We’ll be available if you need support, happy hacking!


## Disclaimer
This respository is part of the Zurich hackathon of [Swiss {ai} Week](https://swiss-ai-weeks.ch/) happening on 26/27 September 2025.

By accessing or using the data provided, you agree to the following terms and conditions.

## Terms and Conditions
> The data is provided solely for the purpose of participating in the hackathon event held in Zurich, Switzerland, and for developing solutions directly related to the specific challenge you have selected. You are strictly prohibited from using the Data for any other purpose, including but not limited to:
> - Commercial use.
> - Research or development outside the scope of this hackathon challenge.
> - Personal use or any other unauthorized activities.
> 
> The data is provided "as is" without any warranties, express or implied, including but not limited to, warranties of merchantability, fitness for a particular purpose, or non-infringement. The hackathon organizers do not guarantee the accuracy, completeness, or reliability of the data.
>
> Immediately following the conclusion of the hackathon event, you are obligated to permanently and securely delete all copies of the data, including any derived or processed data, from all your devices, storage media, and systems. 

## Source of Data
The data of this respository has been provided by [Swisscom](https://www.swisscom.ch/) submitting the challenge.

