import pandas as pd
import os

colunas_bonito = ['Perda (t)', 'Acurácia (t)', 'Precisão (t)', 'Recall (t)', 'Perda (v)', 'Acurácia (v)', 'Precisão (v)', 'Recall (v)']
colunas_raw = ['loss','categorical_accuracy','precision','recall','val_loss','val_categorical_accuracy','val_precision','val_recall']

colunas_map = {
    'loss':'Perda (t)',
    'categorical_accuracy':'Acurácia (t)',
    'precision':'Precisão (t)',
    'recall':'Recall (t)',
    'val_loss':'Perda (v)',
    'val_categorical_accuracy':'Acurácia (v)',
    'val_precision':'Precisão (v)',
    'val_recall':'Recall (v)'
}

if __name__ == '__main__':
    csv_saida_path = os.path.join('files','results','relatorio_final')
    pd_saida = pd.DataFrame(columns=colunas_bonito);
    for i in range(1, 24):
        raw_file_name = os.path.join('files','results',f'exp{i}','history.csv')
        raw_df = pd.read_csv(raw_file_name)
        raw_df.rename(columns=colunas_map, inplace=True)
        row = raw_df.iloc[-1][colunas_bonito]
        pd_saida = pd.concat([pd_saida, pd.DataFrame([row])], ignore_index=True)
    pd_saida.to_csv(csv_saida_path, index=False)
