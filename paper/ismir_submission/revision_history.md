Revision 1
==========

The manuscript is updated based on the kind and constructive inputs from reviewers & meta-reviewer. We tried to reflect on those points as much as possible. Below we re-cap what is done, the concerns, and further what will be done.


## Release notes

- 1. [x] the clarification/definition of interpretability.
    - it was pointed out that it might be unfair to say that the DL model completely lacks and BN models would be better in model interpretability
    - In general, we admit it, so we added a new paragraph in response to it and re-phrased a couple of statements to incorporate it. (i.e., the second paragraph and the last sentence in the third paragraph in `Introduction`)

- 2. [?] the explicit comparison between the DL and BN models via the qualitative interpretability assessment.
    - we agree that the particular method we applied to explore the interpretability of the BN model is also partly applicable for any latent variable involved models, such as DL models. Moreover, we also expect that the DL model could show a comparable outcome.
    - we would like to clarify that we did not intend that the exploratory routine is not a formal and definitive procedure to assess the interpretability of the models. Instead, we hoped it was communicated as providing a qualitative overview of the introduced BN model. As noted in the introduction, the HDPGMM models' construction well states how the model works in a (relatively) humanly understandable manner. Thus, knowing each component's rough meaning already explains how the entire generative process works. At the same time, it is not sufficient in the case of DL models (i.e., knowing the meaning of a latent variable in a particular layer is not as explainable as knowing the meaning of the components in HDPGMM).
    - nevertheless, technically speaking, we could present the explicit comparison between DL and BN for a subset of analyses.
    Unfortunately, we were not allowed to have the necessary space for addressing the above considerations in the main body of the text.
    - we decided to leave the comment here so that readers can check, and we may be able to present the comparison in the web-based supplementary material shortly (i.e., Github)

- 3. minor/editorial issues
    - [x] we swapped the order of the presence of tables 3 and 4 for the better readability
    - [x] we added the `Measure` column in table 3 for the better readability
    - [x] we moved a handful of footnotes into the main text
    - [x] we stripped the `URL` fields in each of the `.bibtex` entries for the compact representation
