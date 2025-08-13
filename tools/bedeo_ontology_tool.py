"""
BEDEO Ontology Tool - Load and parse BEDEO ontology for agents to use
"""
import os
from typing import Dict, List, Set
import re

def load_bedeo_ontology() -> Dict[str, any]:
    """
    Load and parse the BEDEO ontology from bedeo.ttl file.
    
    Returns:
        Dictionary containing:
        - classes: List of BEDEO classes
        - object_properties: List of object properties
        - data_properties: List of data properties
        - example_template: RDF/Turtle template using BEDEO vocabulary
    """
    ontology_path = os.path.join(os.path.dirname(__file__), '../ontology/bedeo.ttl')
    
    if not os.path.exists(ontology_path):
        raise FileNotFoundError(f"BEDEO ontology file not found at {ontology_path}")
    
    with open(ontology_path, 'r') as f:
        content = f.read()
    
    # Parse classes
    classes = set()
    class_pattern = r'bedeo:(\w+)\s+rdf:type\s+owl:Class'
    for match in re.finditer(class_pattern, content):
        classes.add(f"bedeo:{match.group(1)}")
    
    # Parse object properties
    object_properties = set()
    prop_pattern = r'bedeo:(\w+)\s+rdf:type\s+owl:ObjectProperty'
    for match in re.finditer(prop_pattern, content):
        object_properties.add(f"bedeo:{match.group(1)}")
    
    # Parse data properties
    data_properties = set()
    data_prop_pattern = r'bedeo:(\w+)\s+rdf:type\s+owl:DatatypeProperty'
    for match in re.finditer(data_prop_pattern, content):
        data_properties.add(f"bedeo:{match.group(1)}")
    
    # Create example template based on BEDEO ontology
    example_template = """@prefix bedeo: <https://csse.utoronto.ca/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# REQUIRED STRUCTURE: Organization → Opportunity → RealEstateAsset → Address

# The organization offering the opportunity  
bedeo:organization_[OrganizationName]
    a bedeo:Organization ;
    bedeo:has_legal_name "[Organization Legal Name]"^^xsd:string ;
    bedeo:has_opportunity bedeo:opportunity_[OpportunityName] .

# The development opportunity
bedeo:opportunity_[OpportunityName]
    a bedeo:PpartnershipOpportunity ;
    rdfs:label "[Opportunity Description]" ;
    bedeo:has_status "[Status]"^^xsd:string ;
    bedeo:has_real_estate_asset bedeo:realEstateAsset_[AssetName] .

# The real estate asset
bedeo:realEstateAsset_[AssetName]
    a bedeo:real_estate_asset ;
    rdfs:label "[Asset Label]" ;
    bedeo:has_identifier "[Asset Identifier]"^^xsd:string ;
    bedeo:has_surface_area_in_hectares "[Area]"^^xsd:decimal ;
    bedeo:has_address bedeo:address_[AssetName] .

# The address for the asset
bedeo:address_[AssetName]
    a bedeo:Address ;
    rdfs:label "[Address Label]" ;
    bedeo:has_street_number "[Street Number]"^^xsd:string ;
    bedeo:has_street_name "[Street Name]"^^xsd:string ;
    bedeo:has_locality_name "[City]"^^xsd:string ;
    bedeo:has_province_name "[Province]"^^xsd:string ;
    bedeo:has_country_name "[Country]"^^xsd:string ."""
    
    return {
        'classes': sorted(list(classes)),
        'object_properties': sorted(list(object_properties)),
        'data_properties': sorted(list(data_properties)),
        'example_template': example_template,
        'total_classes': len(classes),
        'total_properties': len(object_properties) + len(data_properties)
    }

def get_bedeo_classes() -> List[str]:
    """Get list of all BEDEO classes."""
    ontology = load_bedeo_ontology()
    return ontology['classes']

def get_bedeo_properties() -> Dict[str, List[str]]:
    """Get all BEDEO properties organized by type."""
    ontology = load_bedeo_ontology()
    return {
        'object_properties': ontology['object_properties'],
        'data_properties': ontology['data_properties']
    }

def validate_rdf_against_bedeo(rdf_content: str) -> Dict[str, any]:
    """
    Validate that RDF content uses only BEDEO vocabulary.
    
    Args:
        rdf_content: RDF/Turtle content to validate
        
    Returns:
        Dictionary with validation results
    """
    ontology = load_bedeo_ontology()
    valid_terms = set(ontology['classes'] + ontology['object_properties'] + ontology['data_properties'])
    
    # Find all bedeo: terms in the content
    used_terms = set()
    term_pattern = r'bedeo:(\w+)'
    for match in re.finditer(term_pattern, rdf_content):
        used_terms.add(f"bedeo:{match.group(1)}")
    
    # Check for invalid terms
    invalid_terms = []
    for term in used_terms:
        if term not in valid_terms and not term.startswith('bedeo:organization_') \
           and not term.startswith('bedeo:opportunity_') \
           and not term.startswith('bedeo:realEstateAsset_') \
           and not term.startswith('bedeo:address_'):
            invalid_terms.append(term)
    
    return {
        'is_valid': len(invalid_terms) == 0,
        'invalid_terms': invalid_terms,
        'used_terms': list(used_terms),
        'message': 'Valid BEDEO RDF' if len(invalid_terms) == 0 else f'Invalid terms found: {", ".join(invalid_terms)}'
    }

def get_bedeo_template() -> str:
    """Get the BEDEO RDF template for agents to follow."""
    ontology = load_bedeo_ontology()
    return ontology['example_template']