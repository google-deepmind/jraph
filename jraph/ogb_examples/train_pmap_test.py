# Copyright 2020 DeepMind Technologies Limited.


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for jraph.ogb_examples.train_pmap."""

import pathlib
from absl.testing import absltest
from jraph.ogb_examples import train_pmap


class TrainTest(absltest.TestCase):

  def test_train_and_eval_overfit(self):
    ogb_path = pathlib.Path(train_pmap.__file__).parents[0]
    master_csv_path = pathlib.Path(ogb_path, 'test_data', 'master.csv')
    split_path = pathlib.Path(ogb_path, 'test_data', 'train.csv.gz')
    data_path = master_csv_path.parents[0]
    temp_dir = self.create_tempdir().full_path
    train_pmap.train(data_path, master_csv_path, split_path, 1, 101, temp_dir)
    _, accuracy = train_pmap.evaluate(data_path, master_csv_path, split_path,
                                      temp_dir)
    self.assertEqual(float(accuracy), 1.0)


if __name__ == '__main__':
  absltest.main()
