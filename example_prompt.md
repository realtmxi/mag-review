# Example Prompt for Correct RDF Output

When calling the web crawling agent, provide your instructor's ideal RDF as a few-shot example:

```
I need you to crawl this URL: https://www.toronto.ca/community-people/housing-shelter/affordable-housing-developments/map-of-affordable-housing-locations/35-bellevue-ave/

Use this exact RDF/Turtle ontology pattern as your template:

**Few-Shot Example:**
```turtle
@prefix bedeo: <http://example.org/bedeo#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# The organization offering the opportunity  
bedeo:organization_CanadaLandsCompany
    a bedeo:Organization ;
    bedeo:has_legal_name "Canada Lands Company"^^xsd:string ;
    bedeo:has_opportunity bedeo:opportunity_CurrieDevelopment .

# The development opportunity
bedeo:opportunity_CurrieDevelopment
    a bedeo:PpartnershipOpportunity ;
    rdfs:label "Development opportunity for 99+ units in Currie, Calgary" ;
    bedeo:has_status "Proposals under evaluation"^^xsd:string ;
    bedeo:has_real_estate_asset bedeo:realEstateAsset_CurrieLot .

# The real estate asset
bedeo:realEstateAsset_CurrieLot
    a bedeo:real_estate_asset ;
    rdfs:label "Currie Lot" ;
    bedeo:has_identifier "CLC_LH_AB_CGY_L002"^^xsd:string ;
    bedeo:has_surface_area_in_hectares "0.4"^^xsd:decimal ;
    bedeo:has_address bedeo:address_CurrieLot .

# The address
bedeo:address_CurrieLot
    a bedeo:Address ;
    rdfs:label "Currie Lot Address" ;
    bedeo:has_locality_name "Calgary"^^xsd:string ;
    bedeo:has_province_name "Alberta"^^xsd:string ;
    bedeo:has_country_name "Canada"^^xsd:string .
```

Follow this exact pattern: Organization → Opportunity → RealEstateAsset → Address, using the same property names and bedeo: namespace.
```

This should make the agent generate RDF that matches your instructor's requirements exactly.