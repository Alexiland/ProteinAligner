

# ProteinAligner
ProteinAligner es un framework de representación multimodal de proteínas que unifica datos estructurales, secuenciales y textuales en un único espacio de representación, permitiendo diversas aplicaciones con un entrenamiento mínimo mediante el uso de codificadores de modalidades preentrenados.

Consulte [nuestro manuscrito](https://www.biorxiv.org/content/10.1101/2024.10.06.616870v1.full.pdf) para más detalles.

# Instalación
Para instalarlo, ejecute el siguiente script de bash.
```bash
conda create -n proteinaligner python=3.10
conda activate proteinaligner
pip install -r requirements.txt
```
Tenga en cuenta que, por simplicidad, hemos incluido el ESM dentro de la carpeta pretrain.

# Arquitectura del Modelo y Preentrenamiento
ProteinAligner es un framework innovador de representación multimodal de proteínas que integra datos estructurales, secuenciales y textuales en un espacio de incrustación unificado. Utiliza tres vías de codificación distintas para la secuencia de aminoácidos, la estructura 3D y la descripción textual de una proteína, empleando la secuencia como modalidad de anclaje para la alineación. La vía de la secuencia procesa la secuencia unidimensional de aminoácidos, generando representaciones que capturan las características moleculares de cada aminoácido mediante el modelo de lenguaje de proteínas ESM-2. La vía de la estructura, que utiliza el modelo ESM-IF1, procesa la estructura tridimensional, capturando las interacciones moleculares y la dinámica de la proteína. La vía textual emplea un codificador Transformer de 8 capas estándar para procesar descripciones textuales derivadas de publicaciones verificadas experimentalmente, creando representaciones únicas para cada proteína. Para garantizar la compatibilidad entre modalidades, cada entrada codificada se proyecta a la misma dimensión.

El framework se entrena en un conjunto de datos a gran escala de 150.000 triples (estructura, secuencia, descripción), obtenidos de las bases de datos UniProtKB/Swiss-Prot y RCSB PDB. El conjunto de datos se ensambló mapeando los identificadores de PDB a los de UniProt para recuperar los triples de la misma proteína. Durante la etapa de preentrenamiento, ProteinAligner busca minimizar la pérdida contrastiva entre pares de secuencia-estructura y secuencia-texto, alineando las incrustaciones de las estructuras y las descripciones textuales con las incrustaciones de la secuencia para formar una representación unificada. Esta etapa de preentrenamiento aprovecha la secuencia como modalidad de anclaje para facilitar este proceso de alineación.

El entrenamiento de ProteinAligner sigue un pipeline de dos etapas: preentrenamiento y ajuste fino (fine-tuning). Inicialmente, la etapa de preentrenamiento implica calcular la pérdida contrastiva en datos emparejados por secuencia, centrándose específicamente en pares de secuencia-estructura y secuencia-texto. Este enfoque de aprendizaje contrastivo garantiza que las incrustaciones de las estructuras proteicas y las descripciones textuales se alineen con las incrustaciones de la secuencia de la misma proteína. Posteriormente, la etapa de ajuste fino integra los pesos del codificador preentrenado con capas específicas para la tarea, permitiendo la aplicación de ProteinAligner a una variedad de tareas específicas del dominio. Este enfoque estructurado permite a ProteinAligner aprovechar las fortalezas de cada modalidad, creando un framework robusto y versátil para la representación de proteínas.

Detallamos el modelo, los procedimientos de entrenamiento y el acceso a los datos en [nuestro manuscrito](https://www.biorxiv.org/content/10.1101/2024.10.06.616870v1.full.pdf). 

## Ejecución del código de preentrenamiento
Puede ejecutar el código de preentrenamiento con el siguiente script de bash.

```bash
cd pretrain
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:23308 --nnodes=1 --rdzv_id 234 --nproc-per-node=8 train_joint_encoder.py \
 --output_dir /PATH/TO/SAVE \
 --log_dir /PATH/TO/SAVE \
 --world_size 8
```

# Tareas downstream
Puede consultar [downstream](ProteinAligner_downstream) para ver cómo aplicar ProteinAligner preentrenado a tareas downstream.
