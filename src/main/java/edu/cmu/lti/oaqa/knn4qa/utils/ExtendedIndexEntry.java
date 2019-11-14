/*
 *  Copyright 2014+ Carnegie Mellon University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package edu.cmu.lti.oaqa.knn4qa.utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * An extended index entry, which keeps a dictionary
 * whose elements are either strings or arrays of strings.
 * 
 * @author Leonid Boytsov
 *
 */
public class ExtendedIndexEntry {

  public Map<String, String>    mStringDict = new HashMap<String, String>();
  public Map<String, ArrayList<String>> mStringArrDict = new  HashMap<String, ArrayList<String>>();

}
