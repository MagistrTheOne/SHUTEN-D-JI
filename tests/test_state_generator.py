"""Unit tests for the State Generator."""

import pytest
from src.factory.state_generator import (
    StateGenerator,
    StateGeneratorConfig,
    WorldState,
    Domain,
    EntityType,
)


@pytest.fixture
def generator():
    config = StateGeneratorConfig(seed=42)
    return StateGenerator(config)


def test_generate_single_state(generator):
    state = generator.generate()
    assert isinstance(state, WorldState)
    assert state.id is not None
    assert state.domain in Domain
    assert 1 <= state.complexity_level <= 5
    assert len(state.entities) >= 3


def test_generate_specific_domain(generator):
    state = generator.generate(domain=Domain.BUSINESS)
    assert state.domain == Domain.BUSINESS


def test_generate_specific_complexity(generator):
    state = generator.generate(complexity=4)
    assert state.complexity_level == 4
    assert len(state.entities) > 10  # high complexity = more entities


def test_entities_have_valid_types(generator):
    state = generator.generate()
    for entity in state.entities:
        assert entity.type in EntityType
        assert entity.name != ""
        assert entity.id != ""


def test_relationships_reference_existing_entities(generator):
    state = generator.generate()
    entity_ids = {e.id for e in state.entities}
    for rel in state.relationships:
        assert rel.source_id in entity_ids
        assert rel.target_id in entity_ids
        assert 0 <= rel.strength <= 1


def test_generate_batch(generator):
    states = generator.generate_batch(5)
    assert len(states) == 5
    assert all(isinstance(s, WorldState) for s in states)


def test_deterministic_with_seed():
    gen1 = StateGenerator(StateGeneratorConfig(seed=123))
    gen2 = StateGenerator(StateGeneratorConfig(seed=123))
    state1 = gen1.generate(domain=Domain.MARKETS)
    state2 = gen2.generate(domain=Domain.MARKETS)
    assert len(state1.entities) == len(state2.entities)
    assert state1.complexity_level == state2.complexity_level


def test_all_domains_produce_valid_states(generator):
    for domain in Domain:
        state = generator.generate(domain=domain)
        assert state.domain == domain
        assert len(state.entities) >= 1
