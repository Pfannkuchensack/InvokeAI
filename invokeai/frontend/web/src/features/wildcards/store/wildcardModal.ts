import { atom } from 'nanostores';

const initialState: WildcardModalState = {
  isModalOpen: false,
  updatingWildcardId: null,
  prefilledFormData: null,
};

/**
 * Tracks the state for the wildcard modal.
 */
export const $wildcardModalState = atom<WildcardModalState>(initialState);

type WildcardModalState = {
  isModalOpen: boolean;
  updatingWildcardId: string | null;
  prefilledFormData: PrefilledFormData | null;
};

type PrefilledFormData = {
  name: string;
  values: string[];
  is_public: boolean;
};
