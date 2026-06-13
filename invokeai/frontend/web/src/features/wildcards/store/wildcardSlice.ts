import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import { atom } from 'nanostores';
import { assert } from 'tsafe';
import z from 'zod';

const zWildcardState = z.object({
  searchTerm: z.string(),
});

type WildcardState = z.infer<typeof zWildcardState>;

const getInitialState = (): WildcardState => ({
  searchTerm: '',
});

const slice = createSlice({
  name: 'wildcard',
  initialState: getInitialState(),
  reducers: {
    wildcardSearchTermChanged: (state, action: PayloadAction<string>) => {
      state.searchTerm = action.payload;
    },
  },
});

export const { wildcardSearchTermChanged } = slice.actions;

export const wildcardSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zWildcardState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        state._version = 1;
      }
      return zWildcardState.parse(state);
    },
    persistDenylist: ['searchTerm'],
  },
};

const selectWildcardSlice = (state: RootState) => state.wildcard;
const createWildcardSelector = <T>(selector: Selector<WildcardState, T>) =>
  createSelector(selectWildcardSlice, selector);

export const selectWildcardSearchTerm = createWildcardSelector((wildcard) => wildcard.searchTerm);

/**
 * Tracks whether or not the wildcard menu is open.
 */
export const $isWildcardsMenuOpen = atom<boolean>(false);
