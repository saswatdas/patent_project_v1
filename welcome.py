import streamlit as st


st.set_page_config(
    page_title="Welcome!!!",
    page_icon="👋",
)

st.write("# Welcome to PatentGuru! 👋")
st.sidebar.success("Select a demo above")

st.markdown(
    """
    PatentGuru is an patent search assistant built specifically for
    building patents on any subject. The project uses data from Google patents covering a gamut of 1.3 mn granted patents
    across all categories

    **👈 Select a demo from the dropdown on the left** to see some examples
    of what PatentGuru can do!

    ### Want to learn more about patents?

    - Check out [United States Patent and Trademark office](https://www.uspto.gov/)
    - Jump into the Open Source Documentation that was used for embedding [AI-Growth-Lab](https://huggingface.co/AI-Growth-Lab/PatentSBERTa)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)

    ### See more on tensor flow datasets

    - Use a tensor flow dataset to [analyze the patent categorization and abstract
        Dataset](hhttps://www.tensorflow.org/datasets/catalog/big_patent)
    - Explore a [European Patent Office](https://www.epo.org/en)
"""
    )

