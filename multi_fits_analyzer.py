import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from datetime import datetime
import os

# =================================================================
# SISTEMA AVANZADO PICASER: ANÁLISIS MULTI-TEMPORAL DE IMÁGENES FITS
# Investigadora Principal: Pilar López Giménez
# =================================================================

def analizador_multitemporal_picaser(fits_file_paths):
    """
    Realiza un análisis diferencial temporal comparando un archivo FITS de referencia
    con una lista de archivos FITS posteriores.

    Args:
        fits_file_paths (list): Una lista de rutas a archivos FITS. El primer archivo
                                 se usa como referencia, los demás se comparan con él.

    Returns:
        list: Una lista de diccionarios, donde cada diccionario contiene:
              - 'reference_file': Ruta del archivo de referencia.
              - 'comparison_file': Ruta del archivo de comparación actual.
              - 'diferencial': El mapa diferencial numpy array.
              - 'lista_total': DataFrame de todos los cambios detectados.
              - 'lista_prometedores': DataFrame de los candidatos de alta prioridad.
              - 'error': Mensaje de error si ocurre alguno durante la comparación.
    """
    all_comparisons_results = []

    if not fits_file_paths or len(fits_file_paths) < 2:
        print("❌ Se necesitan al menos dos archivos FITS para el análisis multitemporal.")
        return all_comparisons_results

    reference_path = fits_file_paths[0]
    try:
        data_referencia = fits.open(reference_path)[0].data.astype(float)
        print(f"✅ Archivo de referencia cargado: {reference_path}")
    except Exception as e:
        print(f"❌ Error al cargar el archivo de referencia {reference_path}: {e}")
        return all_comparisons_results

    for i in range(1, len(fits_file_paths)):
        comparison_path = fits_file_paths[i]
        print(f"\n📡 Comparando '{reference_path}' con '{comparison_path}'...")
        comparison_result = {
            'reference_file': reference_path,
            'comparison_file': comparison_path,
            'diferencial': None,
            'lista_total': pd.DataFrame(),
            'lista_prometedores': pd.DataFrame(),
            'error': None
        }

        try:
            data_comparacion = fits.open(comparison_path)[0].data.astype(float)

            # 2. ALINEACIÓN BÁSICA Y LIMPIEZA (Filtro Picaser)
            # Limpiamos el ruido de ambas para comparar solo "señal pura"
            now_clean = data_referencia - gaussian_filter(data_referencia, sigma=15)
            past_clean = data_comparacion - gaussian_filter(data_comparacion, sigma=15)

            # 3. GENERACIÓN DEL DIFERENCIAL (La "Resta Mágica")
            # Lo que sea 0 es que no ha cambiado. Lo que sea > 0 es NUEVO o se ha MOVIDO.
            diferencial = np.abs(now_clean - past_clean)

            umbral = np.std(diferencial) * 5
            puntos = np.where(diferencial > umbral)

            resultados = []

            # 4. CLASIFICACIÓN DE RESULTADOS
            for j in range(len(puntos[0])):
                y, x = puntos[0][j], puntos[1][j]
                intensidad = diferencial[y, x]

                # Un resultado es "Prometedor" si la intensidad es muy alta
                es_prometedor = "SÍ" if intensidad > (umbral * 3) else "No"

                resultados.append({
                    'ID': f"Picaser-Diff-{i}-{j+1}",
                    'Coord_X': x,
                    'Coord_Y': y,
                    'Magnitud_Cambio': round(intensidad, 2),
                    'Prometedor': es_prometedor,
                    'Firma': f"PLG-{datetime.now().year}"
                })

            df_total = pd.DataFrame(resultados)
            df_prometedores = df_total[df_total['Prometedor'] == "SÍ"].sort_values(by='Magnitud_Cambio', ascending=False)

            comparison_result['diferencial'] = diferencial
            comparison_result['lista_total'] = df_total
            comparison_result['lista_prometedores'] = df_prometedores
            print(f"✅ Comparación con '{comparison_path}' completada. Se detectaron {len(df_total)} cambios.")

        except Exception as e:
            comparison_result['error'] = f"Error durante la comparación con {comparison_path}: {e}"
            print(f"❌ {comparison_result['error']}")
        finally:
            all_comparisons_results.append(comparison_result)

    return all_comparisons_results

def generate_picaser_lupas(all_comparisons_results):
    print("Generando visualizaciones detalladas de 'Lupa Picaser' para cada candidato prometedor...")
    lupa_picaser_filenames = []

    for comp_idx, comparison_result in enumerate(all_comparisons_results):
        if comparison_result['error']:
            print(f"Saltando comparación {comp_idx+1} debido a un error: {comparison_result['error']}")
            continue

        # Extraer nombres base para hacer los nombres de archivo únicos
        ref_name = os.path.basename(comparison_result['reference_file']).replace('.FITS', '')
        comp_name = os.path.basename(comparison_result['comparison_file']).replace('.FITS', '')

        diferencial_current = comparison_result['diferencial']
        lista_prometedores_current = comparison_result['lista_prometedores']

        if lista_prometedores_current.empty:
            print(f"No se encontraron candidatos prometedores para la comparación {ref_name} vs {comp_name}.")
            continue

        print(f"Procesando {len(lista_prometedores_current)} candidatos prometedores para la comparación {comp_idx+1}: {ref_name} vs {comp_name}")

        for index, row in lista_prometedores_current.iterrows():
            x, y = int(row['Coord_X']), int(row['Coord_Y'])
            candidate_id = row['ID']

            # Calcula la región de zoom, asegurando que los límites estén dentro de la imagen
            zoom_diff = diferencial_current[max(0, y-15):min(diferencial_current.shape[0], y+15),
                                            max(0, x-15):min(diferencial_current.shape[1], x+15)]

            plt.figure(figsize=(4, 4))
            plt.imshow(zoom_diff, cmap='inferno')
            plt.title(f"HALLAZGO {candidate_id}\n({ref_name} vs {comp_name})")
            plt.axis('off')

            # Crear un nombre de archivo único incluyendo el índice de comparación y los nombres de los archivos
            filename = f'lupa_picaser_comp{comp_idx+1}_{ref_name}_vs_{comp_name}_{candidate_id}.png'
            plt.savefig(filename)
            plt.close() # Cierra la figura para liberar memoria

            lupa_picaser_filenames.append(filename)
            # print(f"  ✅ '{filename}' guardado con éxito.")

    print("Todas las visualizaciones de 'Lupa Picaser' han sido generadas y guardadas.")
    return lupa_picaser_filenames

def generate_multi_comparison_pdf(all_comparisons_results, lupa_picaser_filenames, pdf_output_filename="Informe_Picaser_MultiComparacion.pdf"):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.pagesizes import inch, LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    print(f"Iniciando la generación del informe PDF de Picaser (Multi-Comparación) en '{pdf_output_filename}'...")

    doc_multi = SimpleDocTemplate(pdf_output_filename, pagesize=LETTER)
    story_multi = []
    styles = getSampleStyleSheet()

    story_multi.append(Paragraph("INFORME DEL SISTEMA DE DETECCIÓN PICASER (MULTI-COMPARACIÓN)", styles['h1']))
    story_multi.append(Paragraph("<i>Investigadora Principal: Pilar López Giménez</i>", styles['h2']))
    story_multi.append(Spacer(1, 0.2 * inch))

    intro_text_multi = "Este informe presenta los resultados del análisis diferencial temporal de múltiples comparaciones de imágenes FITS, aplicando el sistema Picaser para la detección de cambios y anomalías. Cada sección detalla una comparación específica, incluyendo su mapa diferencial, tablas de hallazgos y vistas detalladas de los candidatos más prometedores."
    story_multi.append(Paragraph(intro_text_multi, styles['Normal']))
    story_multi.append(Spacer(1, 0.2 * inch))

    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])

    for comp_idx, comparison_result in enumerate(all_comparisons_results):
        if comparison_result['error']:
            story_multi.append(Paragraph(f"<br/><br/><b>ERROR en Comparación {comp_idx+1}: {os.path.basename(comparison_result['reference_file'])} vs {os.path.basename(comparison_result['comparison_file'])}</b>", styles['h2']))
            story_multi.append(Paragraph(f"<i>{comparison_result['error']}</i>", styles['Normal']))
            story_multi.append(Spacer(1, 0.2 * inch))
            continue

        ref_name = os.path.basename(comparison_result['reference_file']).replace('.FITS', '')
        comp_name = os.path.basename(comparison_result['comparison_file']).replace('.FITS', '')

        story_multi.append(Paragraph(f"<br/><br/><b>SECCIÓN {comp_idx+1}: {ref_name} vs {comp_name}</b>", styles['h2']))
        story_multi.append(Spacer(1, 0.1 * inch))

        # Genera y guarda el mapa diferencial principal para esta comparación
        mapa_diferencial_filename = f'mapa_diferencial_comp{comp_idx+1}_{ref_name}_vs_{comp_name}.png'

        plt.figure(figsize=(15, 10))
        plt.imshow(comparison_result['diferencial'], cmap='viridis', origin='lower')
        plt.title(f"MAPA DE DIFERENCIAS TEMPORALES\n({ref_name} vs {comp_name})")
        plt.colorbar(label='Grado de Cambio')

        lista_prometedores_current = comparison_result['lista_prometedores']
        if not lista_prometedores_current.empty:
            sizes = lista_prometedores_current['Magnitud_Cambio'] / lista_prometedores_current['Magnitud_Cambio'].max() * 500
            plt.scatter(lista_prometedores_current['Coord_X'], lista_prometedores_current['Coord_Y'],
                        s=sizes, edgecolors='red', facecolors='none', linewidth=1.5,
                        label='Candidato Prometedor (Tamaño = Magnitud de Cambio)', alpha=0.8)
            plt.legend()
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.savefig(mapa_diferencial_filename)
        plt.close() # Cerrar la figura para liberar memoria

        if os.path.exists(mapa_diferencial_filename):
            available_width = LETTER[0] - 2 * inch
            available_height = LETTER[1] - 3 * inch
            img_mapa_comp = Image(mapa_diferencial_filename, width=available_width, height=available_height)
            story_multi.append(Paragraph("<b>Mapa de Diferencias Temporales</b>", styles['h3']))
            story_multi.append(img_mapa_comp)
            story_multi.append(Spacer(1, 0.2 * inch))
        else:
            story_multi.append(Paragraph(f"<i>Error: Imagen '{mapa_diferencial_filename}' no encontrada.</i>", styles['Normal']))

        story_multi.append(Paragraph("<b>Lista Completa de Cambios Detectados</b>", styles['h3']))
        data_total_for_pdf = [comparison_result['lista_total'].columns.values.tolist()] + comparison_result['lista_total'].values.tolist()
        t_total = Table(data_total_for_pdf)
        t_total._width = LETTER[0] - 2 * inch
        if data_total_for_pdf and data_total_for_pdf[0]: # Check if there are columns
            col_widths = [(LETTER[0] - 2 * inch) / len(data_total_for_pdf[0])] * len(data_total_for_pdf[0])
            t_total._argW = col_widths
        t_total.setStyle(table_style)
        story_multi.append(t_total)
        story_multi.append(Spacer(1, 0.2 * inch))

        story_multi.append(Paragraph("<b>Candidatos Picaser de Alta Prioridad</b>", styles['h3']))
        data_prometedores_for_pdf = [lista_prometedores_current.columns.values.tolist()] + lista_prometedores_current.values.tolist()
        t_prometedores = Table(data_prometedores_for_pdf)
        t_prometedores._width = LETTER[0] - 2 * inch
        if data_prometedores_for_pdf and data_prometedores_for_pdf[0]:
            col_widths = [(LETTER[0] - 2 * inch) / len(data_prometedores_for_pdf[0])] * len(data_prometedores_for_pdf[0])
            t_prometedores._argW = col_widths
        t_prometedores.setStyle(table_style)
        story_multi.append(t_prometedores)
        story_multi.append(Spacer(1, 0.2 * inch))

        story_multi.append(Paragraph("<b>Vistas Detalladas de Lupa Picaser</b>", styles['h3']))

        for index, row in lista_prometedores_current.iterrows():
            candidate_id = row['ID']
            # Construye el nombre del archivo de la lupa basado en cómo se guardó previamente
            lupa_filename = f'lupa_picaser_comp{comp_idx+1}_{ref_name}_vs_{comp_name}_{candidate_id}.png'

            if os.path.exists(lupa_filename):
                story_multi.append(Paragraph(f"<i>Candidato: {candidate_id} - Intensidad: {row['Magnitud_Cambio']:.2f}</i>", styles['Normal']))
                img_lupa_comp = Image(lupa_filename, width=3*inch, height=3*inch)
                story_multi.append(img_lupa_comp)
                story_multi.append(Spacer(1, 0.1 * inch))
            else:
                story_multi.append(Paragraph(f"<i>Error: Imagen '{lupa_filename}' no encontrada para el candidato {candidate_id}.</i>", styles['Normal']))

    doc_multi.build(story_multi)

    print(f"✅ Informe PDF '{pdf_output_filename}' generado con éxito.")

def cleanup_temporary_files(lupa_files, map_files, pdf_file):
    print("Limpiando archivos temporales...")
    all_temp_files = list(lupa_files) + list(map_files)
    for f in all_temp_files:
        if os.path.exists(f):
            os.remove(f)
            # print(f"Removed: {f}")
    # Optionally remove the PDF if there was an error, but usually we want to keep it.
    # if os.path.exists(pdf_file) and error_occurred_during_pdf_gen:
    #     os.remove(pdf_file)
    print("✅ Limpieza de archivos temporales completada.")


if __name__ == "__main__":
    # 1. Entrada de Archivos FITS Interactiva
    fits_file_paths_input = []

    print("\n--- Entrada de Archivos FITS ---")
    print("Recuerda: Los archivos FITS deben estar subidos a Colab o ser accesibles por la ruta completa.")
    print("Introduce el primer archivo FITS (será la referencia para las comparaciones):")

    while True:
        file_path = input(f"Introduce la ruta del archivo FITS {'(o deja vacío para finalizar)' if len(fits_file_paths_input) > 0 else ''}: ")

        if not file_path:
            if len(fits_file_paths_input) < 2:
                print("❌ Se necesitan al menos dos archivos FITS para el análisis multitemporal. Por favor, añade al menos otro archivo.")
                continue
            else:
                break

        if os.path.exists(file_path):
            fits_file_paths_input.append(file_path)
            print(f"'{file_path}' añadido a la lista.")
        else:
            print(f"❌ Error: El archivo '{file_path}' no se encontró. Asegúrate de que la ruta es correcta.")

    print("--- Fin de la Entrada de Archivos ---")
    print(f"Archivos FITS recopilados: {fits_file_paths_input}")

    if len(fits_file_paths_input) < 2:
        print("No hay suficientes archivos para realizar el análisis. Saliendo.")
    else:
        # 2. Ejecutar el análisis multitemporal
        all_comparisons_results = analizador_multitemporal_picaser(fits_file_paths_input)
        print("\nAnálisis multitemporal completado.\n")

        # 3. Generar imágenes de Lupa Picaser
        lupa_picaser_filenames = generate_picaser_lupas(all_comparisons_results)

        # 4. Generar el informe PDF multi-comparación
        final_pdf_name = "Informe_Picaser_MultiComparacion_Final.pdf"
        generate_multi_comparison_pdf(all_comparisons_results, lupa_picaser_filenames, final_pdf_name)

        # 5. Recopilar nombres de los mapas diferenciales generados para limpieza
        mapa_diferencial_filenames = []
        for comp_idx, comparison_result in enumerate(all_comparisons_results):
            if not comparison_result['error']:
                ref_name = os.path.basename(comparison_result['reference_file']).replace('.FITS', '')
                comp_name = os.path.basename(comparison_result['comparison_file']).replace('.FITS', '')
                mapa_diferencial_filenames.append(f'mapa_diferencial_comp{comp_idx+1}_{ref_name}_vs_{comp_name}.png')

        # 6. Limpiar archivos temporales (imágenes PNG)
        cleanup_temporary_files(lupa_picaser_filenames, mapa_diferencial_filenames, final_pdf_name)
