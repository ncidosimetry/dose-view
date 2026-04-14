"""
Patient Data Management Functions

Functions for organizing and managing patient DICOM data.

dicom_json_data is expected to be a dictionary with the following structure:
{
    'patient_id': {
        'sop': [list of SOP Instance UIDs],
        'path': [list of file paths corresponding to the SOPs],
        'ref': [list of reference frame UIDs corresponding to the SOPs],
        'study': [list of study UIDs corresponding to the SOPs],
        'CT': [list of indices in the above lists that correspond to CT images],
        'RTDOSE': [list of indices in the above lists that correspond to RTDOSE images]
    },
    ...
}

"""

from typing import Optional, Dict, Any


def get_patient_data(patient_id: str, dicom_json_data: Dict) -> Optional[Dict[str, Any]]:
    """
    Retrieve patient data from mim_json for a given patient ID.
    
    Parameters
    ----------
    patient_id : str
        The patient ID to search for
    dicom_json_data : dict
        The mim_json dictionary containing patient data
    
    Returns
    -------
    dict or None
        Patient data if found, None if not found
    """
    # First try exact match
    if patient_id in dicom_json_data:
        return dicom_json_data[patient_id]
    
    # If not found, try removing protocol prefix (e.g., "002-" from "002-0002")
    if '-' in patient_id:
        modified_id = patient_id.split('-')[1]
        if modified_id in dicom_json_data:
            return dicom_json_data[modified_id]
    
    # If still not found, try adding common suffixes
    for suffix in ['-002', '-001']:
        test_id = patient_id + suffix
        if test_id in dicom_json_data:
            return dicom_json_data[test_id]
    
    return None


def ct_to_dose(patient_data: Dict) -> Dict:
    """
    Organize CT and RTDOSE files by their reference frame relationships.
    
    Parameters
    ----------
    patient_data : dict
        Dictionary containing patient DICOM data with modality indices
        
    Returns
    -------
    dict
        Dictionary organizing CT to dose relationships
    """
    ct_to_dose_structure = {}
    
    # First, organize by CT
    for ct_idx in patient_data['CT']:
        ct_sop = patient_data['sop'][ct_idx]
        ct_path = patient_data['path'][ct_idx]
        ct_ref = patient_data['ref'][ct_idx]
        ct_study = patient_data['study'][ct_idx]
        
        ct_to_dose_structure[ct_idx] = {
            'ct': {
                'sop': ct_sop,
                'path': ct_path,
                'ref': ct_ref,
                'index': ct_idx,
                'study': ct_study
            },
            'rtdose': [],
            'dose_count': 0
        }
        
        # Find all RTDOSE files that share the same reference frame
        for dose_idx in patient_data['RTDOSE']:
            dose_ref = patient_data['ref'][dose_idx]
            if dose_ref == ct_ref:
                dose_info = {
                    'sop': patient_data['sop'][dose_idx],
                    'path': patient_data['path'][dose_idx],
                    'ref': dose_ref,
                    'index': dose_idx,
                    'study': patient_data['study'][dose_idx]
                }
                ct_to_dose_structure[ct_idx]['rtdose'].append(dose_info)
                ct_to_dose_structure[ct_idx]['dose_count'] += 1

    return ct_to_dose_structure


def dose_to_ct(patient_data: Dict) -> Dict:
    """
    Map RTDOSE files to their corresponding CT images based on reference frames.
    
    Parameters
    ----------
    patient_data : dict
        Dictionary containing patient DICOM data with modality indices
        
    Returns
    -------
    dict
        Dictionary organizing RTDOSE to CT relationships
    """
    
    dose_structure = {} 
    # Create dose-centric view
    for dose_idx in patient_data['RTDOSE']:
        dose_sop = patient_data['sop'][dose_idx]
        dose_path = patient_data['path'][dose_idx]
        dose_ref = patient_data['ref'][dose_idx]
        study = patient_data['study'][dose_idx]
        
        dose_structure[dose_idx] = {
            'dose': {
                'sop': dose_sop,
                'path': dose_path,
                'ref': dose_ref,
                'index': dose_idx,
                'study': study
            },
            'ct': None
        }
        
        # Find the CT that shares the same reference frame
        for ct_idx in patient_data['CT']:
            ct_ref = patient_data['ref'][ct_idx]
            if ct_ref == dose_ref:
                dose_structure[dose_idx]['ct'] = {
                    'sop': patient_data['sop'][ct_idx],
                    'path': patient_data['path'][ct_idx],
                    'ref': ct_ref,
                    'index': ct_idx,
                    'study': patient_data['study'][ct_idx]
                }
                break
                
    return dose_structure
