## aviation_ner_sdr
:rocket: This is the source code to fine-tuning a transformer based model to identify and extract aviation hazards associated with product factors from Service Difficulty Reports.  The work is a collaboration between FAA and Boeing data scientist teams.   The NER model will enhance searchability and facilitate clustering and trend analysis of safety events.

SDRs are submitted via the [service difficulty reporting system](https://sdrs.faa.gov/) operators or certified repair stations as a means to document and share information with the aviation community about failures, malfunctions, or defects of aeronautical products. The free-form text description field often contains valuable COS-related information, however it lacks predictable grammatical structure and is not in any way standardized. Additionally, it can contain typographical errors, part numbers, abbreviations, and references to specific sections of maintenance manuals or operating procedures, making it difficult to reliably extract this information with regular expressions or language models designed to take in clean, full sentences as input. 

Entity	Definition
Flight Phase
(FLT)	It references to the IATA taxonomy that focuses on safety management.  IATA includes flight planning and ground servicing phases since these phases can directly impact a flight.

Product Location
(LOC)	A location within the airplane and directional information which disambiguates one Product Factor from another or helps to identify each aircraft component specifically

Crew Action
(ACT)	A task which is/was carried out to attempt to resolve/correct a Product Condition excluding maintenance action.  Examples: follow QRH, complied with procedure, run or accomplished procedure, disable/enable systems, turn on/off systems, change state of the airplane or its systems, change flight phase, change flight altitude, etc. Communication related actions such as request, call, notify, and notice are excluded.

Product
(PROD)	Airplane and components/equipment/systems installed on the delivered product. Typically, this means something that you can touch, hold, remove, replace, control or interact with. Examples: tire pressure, software, navigation database, and cabin pressure.

Product Condition
(PCON)	A specific quality, behavior, or situation with regards to a Product Factor or Product Location. Examples: Smoke, Fire, Fumes, Odor, Loss of Aircraft Control, FOD, Fuel Issue, Gear Up Landing, Ground Strike, Jet Blast, Loss of VLOS

Bird strike or Animal strike
(BIRD)	Bird strike or a near miss between an aircraft and wildlife, during high-speed take-off or landing.  For animal strike, an impact/collision between an aircraft and wildlife (Deer, elk, coyote, fox), during high-speed take-off or landing
Emergency or Abnormal Situation (SIT)	An emergency situation is one in which the safety of the aircraft or of persons on board or on the ground is endangered for any reason.  An abnormal situation is one in which it is no longer possible to continue the flight using normal procedures but the safety of the aircraft or persons on board or on the ground is not in danger. Examples: Evacuated, Flight Cancelled/Delayed, Diverted, Executed Go Around / Missed Approach, Inflight Shutdown, Exited Penetrated Airspace, FLC Overrode Automation, FLC Complied with Automation / Advisory, Landed as Precaution, Landed in Emergency Condition, Overcame Equipment Problem, Regained Aircraft Control, Rejected Takeoff, Requested ATC Assistance/Clarification, Returned to Clearance, Returned to Departure Airport, Returned to Gate, Returned to Home, Took Evasive Action


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
