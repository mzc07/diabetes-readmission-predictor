# Diabetic Data & IDS Mapping

![Imagen de ilustración sobre la diabetes](https://static.wixstatic.com/media/8f54f7_89f579b8c5d049bbb8bf33f82bc04fb8~mv2.png/v1/fill/w_399,h_330,al_c,lg_1,q_85/Mesa%20de%20trabajo%206.webp)

Los datasets fueron extraidos del sitio web UC Irvine (UCI)[**Diabetes 130-US Hospital for Years 1999-2008**](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008). Este dataset representa diez años de atención en clínica en 130 hospitales de EE.UU. Incluye más de 50 carácteristicas que representan los resultados del paciente y del hospital.
De acuerdo con la fuente, se cumplieron las siguientes condiciones para asegurarse de la integridad del dataset:
- Es un ingreso hospitalario.
- Se ingreso al sistema cualquier tipo de diabetes como diagnostico.
- La duración de la estancia fue de mínimo 1 día y máximo 4 dias.
- Se realizaron pruebas de laboratorio durante el encuentro.
- Se administraron medicamentos durante el encuentro.

## Composición del dataset
- `diabetic_data.csv`

| encounter_id | patient_nbr | race | gender | ... | diabetesMed | readmitted |
| --- | --- | --- | --- | --- | --- | ---|
| 2278392 | 8222157 | Caucasian | Female | ... | No | NO | 
| 149190 | 55629189 | Caucasian | Female | ... | Yes | >30 |
| ... | ... | ... | ... | ... | ... | ...|
| 443867222 | 175429310 | Caucasian | Male | ... | No | NO |

- `IDS_mapping.csv`
