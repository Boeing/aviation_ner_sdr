## aviation_ner_sdr
:rocket: This is the source code for the SDR Classifier package that classifies potential aviation safety hazards from textual data.  The work is a collaboration between FAA and Boeing data scientist teams, and these models aim to assist analysts in Continued Operational Safety (COS) processes.

SDRs are submitted via the [service difficulty reporting system](https://sdrs.faa.gov/) operators or certified repair stations as a means to document and share information with the aviation community about failures, malfunctions, or defects of aeronautical products. The free-form text description field often contains valuable COS-related information, however it lacks predictable grammatical structure and is not in any way standardized. Additionally, it can contain typographical errors, part numbers, abbreviations, and references to specific sections of maintenance manuals or operating procedures, making it difficult to reliably extract this information with regular expressions or language models designed to take in clean, full sentences as input. 

## Demo
![]()

## Virtual Environment
It is highly recommended to use venv, virtualenv or conda python environments. Read more about creating virtual environments via venv
https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments

## Contributing
🛩️ Please follow the [contribution guideline]()

## Example
:airplane: Follow the code snippet below to test and call the prediction method from the Depressurization model

```
import pandas as pd
(to be added)
```

## References
[Open Access Data via the FAA's Service Difficulty Reporting System](https://sdrs.faa.gov/)

[Full Academic Paper on IEEE Xplore]()
